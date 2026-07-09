# mechanism_diagnosis_standalone.py
import os
# 必须在导入其他库之前强制设置镜像，防止网络报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import tiktoken

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 50257

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# ==================== 本地数据加载 ====================
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

def get_batch(data, bs, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (bs,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

# ==================== 模型定义 (完全复现) ====================
class LearnableScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * 0.1)
    def forward(self, x):
        return x * torch.clamp(self.weight, -2.0, 2.0)

class BilinearMemoryBlockV3(nn.Module):
    def __init__(self, dim, state_dim, memory_slots=128, rank=32,
                 use_bilinear=True, use_memory=True, use_time=True,
                 use_memory_update=False, memory_inject_scale=2.0):
        super().__init__()
        self.dim, self.state_dim, self.memory_slots = dim, state_dim, memory_slots
        self.use_bilinear, self.use_memory, self.use_time = use_bilinear, use_memory, use_time
        self.use_memory_update = use_memory_update
        self.memory_inject_scale = memory_inject_scale

        if use_bilinear:
            self.L_x = nn.Parameter(torch.randn(dim, rank) * 0.02)
            self.R_xs = nn.Parameter(torch.randn(rank, state_dim, dim) * 0.02)
        if use_memory:
            self.W_read = nn.Parameter(torch.randn(dim, memory_slots) * 0.02)
            self.memory_init = nn.Parameter(torch.randn(memory_slots, dim) * 0.02)
        if use_time:
            self.time_alpha = nn.Parameter(torch.tensor(0.3))

        self.W_g = nn.Parameter(torch.randn(dim, state_dim) * 0.02)
        self.norm_x = LearnableScale(dim)
        self.norm_state = LearnableScale(state_dim)

    def forward(self, x, state, memory=None):
        B, T, D = x.shape
        S, N = self.state_dim, self.memory_slots

        if self.use_time:
            x_time = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
            x = x + self.time_alpha * x_time

        xf = x.reshape(B * T, D)
        inter_s = 0.0
        if self.use_bilinear:
            x_proj = xf @ self.L_x
            state_flat = state.reshape(B * T, S)
            temp = torch.einsum('bs,rsd->brd', state_flat, self.R_xs)
            inter_s = torch.einsum('br,brd->bd', x_proj, temp)
            inter_s = torch.clamp(inter_s, -5.0, 5.0)

        mem_ctx = 0.0
        if self.use_memory:
            gate_read = torch.tanh(xf @ self.W_read)
            mem_ctx = gate_read.reshape(B, T, N) @ self.memory_init
            mem_ctx = mem_ctx.reshape(B * T, D)
            mem_ctx = torch.clamp(mem_ctx, -5.0, 5.0)

        x_new = xf + inter_s + self.memory_inject_scale * mem_ctx
        x_new = torch.clamp(x_new, -10.0, 10.0).reshape(B, T, D)

        gate = torch.tanh(xf @ self.W_g).reshape(B, T, S)
        state_new = gate * x_new + (1 - gate) * state
        state_new = torch.clamp(state_new, -10.0, 10.0)

        x_out = self.norm_x(x + x_new)
        state_out = self.norm_state(state + state_new)
        return x_out, state_out, None

class BMM(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, state_dim=768, num_blocks=8,
                 memory_slots=128, rank=32, max_pos=2048, **block_kwargs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_phase = nn.Parameter(torch.randn(1, max_pos, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            BilinearMemoryBlockV3(embed_dim, state_dim, memory_slots, rank, **block_kwargs)
            for _ in range(num_blocks)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        pos_idx = torch.arange(T, device=idx.device) % self.pos_phase.shape[1]
        x = self.embed(idx) + self.pos_phase[:, pos_idx, :]
        state = torch.zeros(B, T, self.blocks[0].state_dim, device=x.device)
        memory = None
        for blk in self.blocks:
            x, state, memory = blk(x, state, memory)
        return self.head(x)

class BMM_Tied(BMM):
    def __init__(self, vocab_size, embed_dim=768, state_dim=768, num_blocks=8,
                 memory_slots=128, rank=32, max_pos=2048, **block_kwargs):
        super().__init__(vocab_size, embed_dim, state_dim, num_blocks, memory_slots, rank, max_pos, **block_kwargs)
        self.head.weight = self.embed.weight
        if self.head.bias is not None:
            self.head.bias.requires_grad = False
            self.head.bias.data.zero_()

# ==================== 诊断主流程 ====================
def main():
    print("Loading BMM model for diagnosis...")
    bmm = BMM_Tied(vocab_size=VOCAB_SIZE, embed_dim=768, state_dim=768, num_blocks=8,
                   memory_slots=128, rank=32, max_pos=2048,
                   use_bilinear=True, use_memory=True, use_time=True, 
                   use_memory_update=False, memory_inject_scale=2.0).to(DEVICE)
    bmm.load_state_dict(torch.load("bmm_bpe_gamma2.0.pt", map_location=DEVICE))
    bmm.eval()

    _, val_data = prepare_data_bpe()

    # ----- 实验 1: Memory Slots 语义解析 -----
    print("\n===== Exp 1: Memory Slots Semantic Analysis =====")
    memory = bmm.blocks[0].memory_init.detach().cpu().numpy()
    sim_matrix = cosine_similarity(memory)
    head_weight = bmm.head.weight.detach().cpu().numpy()
    token_sim = cosine_similarity(memory, head_weight)
    enc = tiktoken.get_encoding("gpt2")
    
    print("Top matched tokens for each memory slot (sample):")
    for i in [0, 10, 20, 50, 100, 127]:
        top_5_idx = np.argsort(token_sim[i])[-5:][::-1]
        top_5_tokens = [enc.decode([idx]) for idx in top_5_idx]
        print(f"  Slot {i:3d}: {top_5_tokens}")
        
    plt.figure(figsize=(8, 8))
    plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    plt.title('Memory Slots Cosine Similarity Matrix')
    plt.xlabel('Memory Slot Index')
    plt.ylabel('Memory Slot Index')
    plt.tight_layout()
    plt.savefig('memory_slots_similarity.pdf', dpi=300)
    print("Saved memory_slots_similarity.pdf")

    # ----- 实验 2: Temporal Loop 层级分析 -----
    print("\n===== Exp 2: Temporal Loop Alpha Analysis =====")
    alphas = []
    for i, blk in enumerate(bmm.blocks):
        alpha = blk.time_alpha.item()
        alphas.append(alpha)
        print(f"  Layer {i+1}: time_alpha = {alpha:.4f}")
        
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(alphas)+1), alphas, color='skyblue')
    plt.xlabel('Layer Depth', fontsize=12)
    plt.ylabel('Temporal Loop Alpha', fontsize=12)
    plt.title('Temporal Loop Alpha Across Layers')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('temporal_alpha_distribution.pdf', dpi=300)
    print("Saved temporal_alpha_distribution.pdf")

    # ----- 实验 3: 长序列状态诊断 -----
    print("\n===== Exp 3: Long Sequence State Diagnosis =====")
    lengths = [512, 2048, 8192]
    norms = []
    
    for l in lengths:
        print(f"  Testing sequence length: {l}...")
        xv, _ = get_batch(val_data, 1, l, DEVICE)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
            state_norms = []
            def hook_fn(module, input, output):
                state_out = output[1]
                norm = torch.norm(state_out, p=2, dim=-1).mean().item()
                state_norms.append(norm)
                
            handle = bmm.blocks[-1].register_forward_hook(hook_fn)
            _ = bmm(xv)
            handle.remove()
            
            avg_norm = np.mean(state_norms)
            norms.append(avg_norm)
            print(f"    Avg State L2 Norm: {avg_norm:.4f}")
            
    plt.figure(figsize=(6, 4))
    plt.plot(lengths, norms, 'ro-', markersize=8, linewidth=2)
    plt.xscale('log')
    plt.xlabel('Sequence Length (Log Scale)', fontsize=12)
    plt.ylabel('Avg L2 Norm of State', fontsize=12)
    plt.title('State Activation Magnitude vs. Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('state_norm_vs_length.pdf', dpi=300)
    print("Saved state_norm_vs_length.pdf")

    print("\n===== Diagnosis Complete =====")

if __name__ == "__main__":
    main()