# run_ablations_gamma2.py
import os, time, math, random
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from bmm_model import BMM, prepare_data_bpe, get_batch

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
STEPS = 10000
WARMUP = 1000
LR = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 0.5

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

def train_ablation(config_name, **block_kwargs):
    reset_seed()
    print(f"\n===== Ablation: {config_name} =====")
    train_data, val_data, _, _ = prepare_data_bpe()
    
    model = BMM_Tied(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, state_dim=STATE_DIM, 
        num_blocks=NUM_BLOCKS, memory_slots=MEMORY_SLOTS, rank=RANK, max_pos=2048,
        **block_kwargs
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

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
    print(f"===== {config_name} Complete. Final PPL: {final_ppl:.2f} =====")
    return final_ppl

def main():
    results = {}
    # 基准更新为 γ=2.0
    results['Full_Model'] = train_ablation("Full Model (γ=2.0)", use_bilinear=True, use_memory=True, use_time=True, memory_inject_scale=2.0)
    results['No_Temporal_Loop'] = train_ablation("No Temporal Loop", use_time=False, memory_inject_scale=2.0)
    results['No_Bilinear'] = train_ablation("No Bilinear", use_bilinear=False, memory_inject_scale=2.0)
    results['No_Memory'] = train_ablation("No Memory", use_memory=False, memory_inject_scale=0.0)
    results['Gamma_0.0'] = train_ablation("Gamma 0.0", memory_inject_scale=0.0)
    
    print("\n\n===== ABLATION SUMMARY (10k steps, γ=2.0 baseline) =====")
    print(f"{'Configuration':<25} | {'Val PPL':<10}")
    print("-" * 40)
    for config, ppl in results.items():
        print(f"{config:<25} | {ppl:<10.2f}")

if __name__ == "__main__":
    main()