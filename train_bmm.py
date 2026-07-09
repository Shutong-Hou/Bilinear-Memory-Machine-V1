# train_bmm_final_50k.py
import os, time, math, random
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from bmm_model import BMM, prepare_data_bpe, get_batch

# ==================== 全局配置 ====================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 50257
EMBED_DIM = 768
STATE_DIM = 768
NUM_BLOCKS = 8
MEMORY_SLOTS = 128
RANK = 32
BATCH_SIZE = 8
SEQ_LEN = 512
STEPS = 50000
WARMUP = 2000
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 0.5
GAMMA = 2.0  # 锁定安全且高效的 Gamma

class BMM_Tied(BMM):
    def __init__(self, vocab_size, embed_dim=768, state_dim=768, num_blocks=8,
                 memory_slots=128, rank=32, max_pos=2048, **block_kwargs):
        super().__init__(vocab_size, embed_dim, state_dim, num_blocks, memory_slots, rank, max_pos, **block_kwargs)
        self.head.weight = self.embed.weight
        if self.head.bias is not None:
            self.head.bias.requires_grad = False
            self.head.bias.data.zero_()

def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

def main():
    reset_seed()
    print(f"===== Final Main Training BMM BPE (γ={GAMMA}) =====")
    print("Preparing BPE data...")
    train_data, val_data, _, _ = prepare_data_bpe()
    
    model = BMM_Tied(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, state_dim=STATE_DIM, 
        num_blocks=NUM_BLOCKS, memory_slots=MEMORY_SLOTS, rank=RANK, max_pos=2048,
        use_bilinear=True, use_memory=True, use_time=True, 
        use_memory_update=False, memory_inject_scale=GAMMA
    ).to(DEVICE)
    
    total = sum(p.numel() for p in model.parameters())
    print(f"BMM BPE (γ={GAMMA}) parameters: {total:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
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
            # 修复：这里是训练时的 mean loss
            train_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
        scaler.scale(train_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step % 1000 == 0 or step == 1:
            model.eval()
            total_eval_loss = 0.0
            total_tokens = 0
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                for _ in range(3):
                    xv, yv = get_batch(val_data, BATCH_SIZE*2, SEQ_LEN, DEVICE)
                    logits = model(xv)
                    # 修复：使用单独的变量名，且用 sum 计算 PPL
                    eval_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yv.view(-1), reduction='sum')
                    total_eval_loss += eval_loss.item()
                    total_tokens += xv.numel()
            
            ppl = math.exp(min(total_eval_loss / total_tokens, 20))
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - t_start
            # 修复：打印正确的 train_loss (平均值，应在 4-10 之间)
            print(f"Step {step:5d}/{STEPS} | Val PPL: {ppl:.2f} | Train Loss: {train_loss.item():.4f} | LR: {current_lr:.2e} | Time: {elapsed:.0f}s")

    total_time = time.time() - t_start
    model.eval()
    total_eval_loss = 0.0
    total_tokens = 0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
        for _ in range(5):
            xv, yv = get_batch(val_data, BATCH_SIZE*2, SEQ_LEN, DEVICE)
            logits = model(xv)
            eval_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yv.view(-1), reduction='sum')
            total_eval_loss += eval_loss.item()
            total_tokens += xv.numel()
    final_ppl = math.exp(total_eval_loss / total_tokens)
    
    print(f"\n===== Final Main Training Complete =====")
    print(f"BPE (γ={GAMMA}) Final PPL: {final_ppl:.2f}, Total Time: {total_time:.1f}s")
    torch.save(model.state_dict(), f"bmm_bpe_gamma{GAMMA}.pt")

if __name__ == "__main__":
    main()