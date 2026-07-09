# train_mamba_bpe_final.py
import os, time, math, random
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
import tiktoken
from bmm_model import get_batch
from mamba_ssm import Mamba

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

VOCAB_SIZE = 50257
SEQ_LEN = 512
BATCH_SIZE = 8
STEPS = 50000
WARMUP = 2000
LR = 2e-4        # 修改：适配 d=1024 的下调学习率
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0  # 修改：放宽梯度裁剪以适配 SSM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data_bpe():
    print("Loading WikiText-103 and encoding with BPE...")
    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    texts = dataset["train"]["text"]
    raw_text = "\n".join([t for t in texts if t.strip() != ""])
    data = enc.encode_ordinary(raw_text)
    data = torch.tensor(data, dtype=torch.long)
    n = int(len(data) * 0.9)
    return data[:n], data[n:]

# 修改：eps 统一为官方默认的 1e-6
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand)
        
    def forward(self, x, inference_params=None):
        return x + self.mamba(self.norm(x), inference_params=inference_params)

class MambaBPE(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_layer=24, d_state=16, expand=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight # Weight Tying
            
    def forward(self, idx, inference_params=None):
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        x = self.norm_f(x)
        return self.head(x)

def main():
    print("Preparing BPE data...")
    train_data, val_data = prepare_data_bpe()
    model = MambaBPE(VOCAB_SIZE).to(DEVICE)
    # 修改：打印时补充关键配置注释
    print(f"Mamba BPE (Final, d_model=1024, expand=2) params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
    torch.backends.cudnn.benchmark = True

    t_start = time.time()
    for step in range(1, STEPS+1):
        if step <= WARMUP:
            for pg in optimizer.param_groups:
                pg['lr'] = LR * step / WARMUP
        else:
            scheduler.step()
            
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step % 1000 == 0 or step == 1:
            model.eval()
            total_loss = 0.0
            total_tokens = 0
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                for _ in range(3):
                    xv, yv = get_batch(val_data, BATCH_SIZE*2, SEQ_LEN, DEVICE)
                    logits = model(xv)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yv.view(-1), reduction='sum')
                    total_loss += loss.item()
                    total_tokens += xv.numel()
            ppl = math.exp(min(total_loss / total_tokens, 20))
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_start
            print(f"Mamba Step {step:5d} | Val PPL: {ppl:.2f} | LR: {current_lr:.2e} | Time: {elapsed:.0f}s")

    total_time = time.time() - t_start
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
        for _ in range(5):
            xv, yv = get_batch(val_data, BATCH_SIZE*2, SEQ_LEN, DEVICE)
            logits = model(xv)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yv.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += xv.numel()
    final_ppl = math.exp(total_loss / total_tokens)
    
    print(f"\n===== Mamba BPE Training Complete =====")
    print(f"Final PPL: {final_ppl:.2f}, Total Time: {total_time:.1f}s")
    torch.save(model.state_dict(), "mamba_bpe_fair.pt")
    
    # 修改：增加日志落地文件
    with open("mamba_bpe_fair_log.txt", "w") as f:
        f.write(f"d_model=1024,expand=2,final_ppl:{final_ppl:.2f},time:{total_time:.1f}s\n")

if __name__ == "__main__":
    main()