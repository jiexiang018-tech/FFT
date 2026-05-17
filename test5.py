import os
import time
import math
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# ==========================================
# 1. 独立自作トークナイザー（ORE-Tokenizer）ビルド関数
# ==========================================

def ensure_ore_tokenizer(tokenizer_json="ore_tokenizer.json"):
    """
    ローカルに ore_tokenizer.json が存在しない場合、
    データセットから自動サンプリングして数字・記号特化型BPEトークナイザーをビルドする。
    """
    if os.path.exists(tokenizer_json):
        print(f"✅ 既存の ORE-Tokenizer を検出しました: {tokenizer_json}")
        return

    print("🎬 [ORE-Tokenizer] 新生カスタムトークナイザー（語彙数32,000・数理孤立化仕様）のビルドを開始します...")
    
    train_txt_path = "tokenizer_train_corpus.txt"
    if not os.path.exists(train_txt_path):
        print("📥 トークナイザー訓練用の高品質コーパスをデータセットからサンプリング中...")
        with open(train_txt_path, "w", encoding="utf-8") as f:
            for name in ["web_samples_v2", "auto_math_text", "stories"]:
                print(f"  - '{name}' からサンプリング中...")
                ds = load_dataset("HuggingFaceTB/cosmopedia", name=name, split="train", streaming=True)
                count = 0
                for item in ds:
                    f.write(item["text"] + "\n")
                    count += 1
                    if count >= 15000:
                        break
                        
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # 数字と主要な数学記号・括弧類を強制隔離するルール
    math_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+|[\p{N}]|[\+\-\*\/=\(\)\{\}\[\]\.,:;]|[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(math_pattern, behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False)
    ])
    tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    print("🔥 BPE（Byte Pair Encoding）アルゴリズムによる語彙最適化を実行中...")
    tokenizer.train([train_txt_path], trainer)
    
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tokenizer.token_to_id("<s>")), ("</s>", tokenizer.token_to_id("</s>"))],
    )
    
    tokenizer.save(tokenizer_json)
    print(f"✨ 完了！ORE独自の知能の種となるトークナイザーが生成されました: {tokenizer_json}")
    
    test_str = "Solve: 1024 + 48 = 1072."
    encoded = tokenizer.encode(test_str)
    print(f"🔍 テストエンコード確認: '{test_str}'")
    print(f"➔ トークン分割結果: {encoded.tokens}\n")
    
    if os.path.exists(train_txt_path):
        os.remove(train_txt_path)


# ==========================================
# 2. アーキテクチャ定義 (v13.0：完全因果的FFT + True-MoE)
# ==========================================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): 
        return x + self.pe[:, :x.size(1)]


class DynamicCausalFFT(nn.Module):
    def __init__(self, dim, kernel_size=1024):
        super().__init__()
        self.dim, self.kernel_size, self.n_fft = dim, kernel_size, 2048
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.base_decay = nn.Parameter(torch.exp(-torch.linspace(0, 5, kernel_size)))
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.bypass_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        kv = (k * v).to(torch.float32)
        
        padded_kv = F.pad(kv, (0, 0, 0, self.kernel_size))  
        src_f = torch.fft.rfft(padded_kv, n=self.n_fft, dim=1)
        
        kernel = torch.zeros(self.n_fft, D, device=x.device, dtype=torch.float32)
        kernel[:self.kernel_size, :] = self.base_decay.view(-1, 1).expand(-1, D)
        kernel_f = torch.fft.rfft(kernel, n=self.n_fft, dim=0)
        
        res_full = torch.fft.irfft(src_f * kernel_f.unsqueeze(0), n=self.n_fft, dim=1)
        res = res_full[:, :L, :]
        
        fft_out = torch.tanh(res.to(x.dtype)) * self.gate(x)
        alpha = torch.sigmoid(self.bypass_alpha)
        
        return self.out_proj(q + fft_out) + alpha * x


class Expert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        h = int(dim * 4 * 2 / 3)  
        self.w1, self.w2, self.w3 = nn.Linear(dim, h, bias=False), nn.Linear(dim, h, bias=False), nn.Linear(h, dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x): 
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class MoELayer(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x, temp=1.0):
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        logits = self.router(x_flat)
        probs = F.softmax(logits / temp, dim=-1)
        
        importance = probs.mean(0)
        load = (probs > (1.0 / self.num_experts)).float().mean(0)
        aux_loss = (importance.std() / (importance.mean() + 1e-4)) + (load.std() / (load.mean() + 1e-4))

        top_weights, top_indices = torch.topk(probs, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-4)

        out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (top_indices == i)
            if mask.any():
                t_idx, k_pos = torch.where(mask)
                expert_out = self.experts[i](x_flat[t_idx])
                out.index_add_(0, t_idx, expert_out * top_weights[t_idx, k_pos].unsqueeze(-1))
        
        return out.view(B, L, D), aux_loss


class ApexLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1, self.resonance, self.norm2 = nn.LayerNorm(dim), DynamicCausalFFT(dim), nn.LayerNorm(dim)
        self.moe = MoELayer(dim, num_experts=4, top_k=2)
        self.gamma1, self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-4), nn.Parameter(torch.ones(dim) * 1e-4)

    def forward(self, x, temp=1.0):
        x = x + self.gamma1 * self.resonance(self.norm1(x))
        moe_out, aux = self.moe(self.norm2(x), temp=temp)
        x = x + self.gamma2 * moe_out
        return x, aux


class ORE_Apex_v13_0(nn.Module):
    def __init__(self, vocab_size, dim=1024, num_layers=3):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_enc = SinusoidalPositionalEncoding(dim, max_len=1024)
        self.layers = nn.ModuleList([ApexLayer(dim) for _ in range(num_layers)])
        self.norm, self.head = nn.LayerNorm(dim), nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight  

    def forward(self, x, temp=1.0):
        x = self.pos_enc(self.token_emb(x))
        total_aux = 0
        for layer in self.layers:
            x, aux = layer(x, temp=temp)
            total_aux += aux
        return self.head(self.norm(x)), total_aux / len(self.layers)


# ==========================================
# 3. メイン学習セッション
# ==========================================

def run_v13_session():
    print("=============================================================")
    print("       🔬 ORE-Apex v13.0 | 100% Pure Custom Engine 🔬       ")
    print("=============================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎮 稼働デバイス: {device}")
    
    tokenizer_json = "ore_tokenizer.json"
    ensure_ore_tokenizer(tokenizer_json)
    
    print("🤖 自作 ORE-Tokenizer をメモリに展開中...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>"
    )
    
    vocab_size = tokenizer.vocab_size
    print(f"📦 最適化語彙サイズ: {vocab_size} (32000スリム仕様・爆速確定)")
    
    print("🚀 高品質複合知能データセット（Cosmopedia）をロード中...")
    try:
        ds_cosmo = load_dataset("HuggingFaceTB/cosmopedia", name="web_samples_v2", split="train", streaming=True)
        ds_math  = load_dataset("HuggingFaceTB/cosmopedia", name="auto_math_text", split="train", streaming=True)
        ds_story = load_dataset("HuggingFaceTB/cosmopedia", name="stories", split="train", streaming=True)
        
        combined_ds = interleave_datasets(
            [ds_cosmo, ds_math, ds_story], 
            probabilities=[0.4, 0.4, 0.2],  
            stopping_strategy="all_exhausted"
        )
    except Exception as e:
        print(f"❌ データセット読み込みエラー: {e}")
        return

    tokenized_ds = combined_ds.map(
        lambda e: tokenizer(e["text"], truncation=True, max_length=1024, padding="max_length"), 
        batched=True
    )

    # 黄金比セッティング
    batch_size = 1        
    grad_accum_steps = 8  
    
    # 🌟【あなたの追加した最終最適化版・最強のcollate_fn】
    # 将来バッチサイズをいくらに上げても、歪みを完璧に吸収して頑強に動作する設計
    def clean_collate_fn(batch):
        cleaned_list = []
        for item in batch:
            raw_ids = item['input_ids']
            
            # ストリーミング特有の二重リスト [ [...] ] の歪みを安全に解除
            if isinstance(raw_ids, list) and len(raw_ids) > 0 and isinstance(raw_ids[0], list):
                raw_ids = raw_ids[0]
                
            cleaned_list.append(torch.tensor(raw_ids, dtype=torch.long))
        
        # 全データをまとめ上げて [Batch, Sequence_Length] の LongTensor テンソルへ
        return {'input_ids': torch.stack(cleaned_list)}
    
    loader = DataLoader(
        tokenized_ds, 
        batch_size=batch_size, 
        collate_fn=clean_collate_fn
    )

    model = ORE_Apex_v13_0(vocab_size, dim=1024, num_layers=3).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 3000, 150000)
    scaler = torch.amp.GradScaler('cuda') 

    model.train()
    step = 0
    start_time = time.time()
    
    print("\n🔥 トレーニングセッション開始！グラボ点火。")
    print("-------------------------------------------------------------")
    try:
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            current_temp = max(1.0, 2.5 - (step / 5000))
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, aux_loss = model(input_ids[:, :-1], temp=current_temp)
                
                raw_main_loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), 
                    input_ids[:, 1:].reshape(-1), 
                    label_smoothing=0.1
                )
                
                scaled_main_loss = raw_main_loss / grad_accum_steps
                scaled_aux_loss = aux_loss / grad_accum_steps
                total_loss = scaled_main_loss + 1.0 * scaled_aux_loss
            
            scaler.scale(total_loss).backward()

            if (i + 1) % grad_accum_steps == 0:
                step += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                if step % 50 == 0:
                    elapsed = time.time() - start_time
                    alpha_vals = [torch.sigmoid(layer.resonance.bypass_alpha).item() for layer in model.layers]
                    alpha_str = ", ".join([f"L{idx}:{v:.2f}" for idx, v in enumerate(alpha_vals)])
                    
                    print(f"Step {step:06d} | Loss: {raw_main_loss.item():.4f} | Aux: {aux_loss.item():.4f} | BypassAlpha ({alpha_str}) | {elapsed:.1f}s")
                
                if step % 2000 == 0:
                    save_name = f"ore_v13_0_step{step}.pt"
                    torch.save(model.state_dict(), save_name)
                    print(f"💾 チェックポイントを保存しました: {save_name}")
                    
                if step >= 150000: 
                    print("🏁 150,000歩の限界突破トレーニングを完走しました！")
                    break

    except KeyboardInterrupt:
        torch.save(model.state_dict(), "ore_v13_0_interrupted.pt")
        print("\n💾 Progress Saved (Interrupted). 中断データが正常に保護されました。")
    except Exception as e:
        print(f"❌ Training Error: {e}")

if __name__ == "__main__":
    run_v13_session()