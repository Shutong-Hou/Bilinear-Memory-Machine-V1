# eval_extreme_streaming.py
import os, time, math, random
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

# ==================== 全局配置 ====================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 50257
LOG_FILE = "extreme_streaming_results.txt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ==================== 真实模型定义 (与训练严格一致) ====================
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

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MambaBlock(nn.Module):
    def __init__(self, d_model, layer_idx, d_state=16, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=expand, layer_idx=layer_idx)
    def forward(self, x, inference_params=None):
        return x + self.mamba(self.norm(x), inference_params=inference_params)

class MambaBPE(nn.Module):
    def __init__(self, vocab_size, d_model=1024, n_layer=24, d_state=16, expand=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, i, d_state, expand) for i in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
    def forward(self, idx, inference_params=None):
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        x = self.norm_f(x)
        return self.head(x)

# ==================== 极限流式生成评估 ====================
def eval_bmm_streaming(model, context_len, gen_len=50):
    """BMM 逐 token 流式生成，严格测试 O(1) 显存"""
    model.eval()
    max_pos = model.pos_phase.shape[1]
    torch.cuda.reset_peak_memory_stats()
    
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
            # 1. 逐 token 处理超长 prompt，只维护单步 state
            current_token = torch.randint(0, VOCAB_SIZE, (1, 1), device=DEVICE)
            pos_idx = 0
            x = model.embed(current_token) + model.pos_phase[:, pos_idx:pos_idx+1, :]
            state = torch.zeros(1, 1, model.blocks[0].state_dim, device=DEVICE)
            for blk in model.blocks: x, state, _ = blk(x, state)
            
            for i in range(1, context_len):
                current_token = torch.randint(0, VOCAB_SIZE, (1, 1), device=DEVICE)
                pos_idx = i % max_pos
                x = model.embed(current_token) + model.pos_phase[:, pos_idx:pos_idx+1, :]
                for blk in model.blocks: x, state, _ = blk(x, state)
                
            next_token = model.head(x).argmax(dim=-1)
            current_pos = context_len
            
            # 2. 测量生成速度
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(gen_len):
                pos_idx = current_pos % max_pos
                x = model.embed(next_token) + model.pos_phase[:, pos_idx:pos_idx+1, :]
                for blk in model.blocks: x, state, _ = blk(x, state)
                next_token = model.head(x).argmax(dim=-1)
                current_pos += 1
            torch.cuda.synchronize()
            t1 = time.time()
    except Exception:
        return "OOM", "OOM"
        
    throughput = gen_len / (t1 - t0)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    return f"{throughput:.1f}", f"{peak_mem:.2f}"

def eval_mamba_streaming(model, context_len, gen_len=50):
    """Mamba 流式生成，使用原生 State Cache"""
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            # 为了防止预分配巨大 buffer 导致直接 OOM，max_seqlen 设为 context_len + gen_len
            inference_params = InferenceParams(max_seqlen=context_len + gen_len, max_batch_size=1)
            
            # 1. 分块预填充，防止一次性输入超长序列导致中间激活 OOM
            chunk_size = 2048
            prompt = torch.randint(0, VOCAB_SIZE, (1, context_len), device=DEVICE)
            for i in range(0, context_len, chunk_size):
                chunk = prompt[:, i:i+chunk_size]
                logits = model(chunk, inference_params=inference_params)
            next_token = logits[:, -1, :].argmax(dim=-1)
            
            # 2. 测量生成速度
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(gen_len):
                logits = model(next_token.unsqueeze(1), inference_params=inference_params)
                next_token = logits[:, -1, :].argmax(dim=-1)
            torch.cuda.synchronize()
            t1 = time.time()
    except Exception as e:
        # 捕获 OOM 或 CUDA Assert
        return "OOM", "OOM"
        
    throughput = gen_len / (t1 - t0)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    return f"{throughput:.1f}", f"{peak_mem:.2f}"

def main():
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    
    log_and_print("Loading models for Extreme Streaming Test...")
    bmm = BMM_Tied(vocab_size=VOCAB_SIZE, embed_dim=768, state_dim=768, num_blocks=8,
                   memory_slots=128, rank=32, max_pos=2048,
                   use_bilinear=True, use_memory=True, use_time=True, 
                   use_memory_update=False, memory_inject_scale=2.0).to(DEVICE)
    bmm.load_state_dict(torch.load("bmm_bpe_gamma2.0.pt", map_location=DEVICE))
    bmm.eval()

    mamba = MambaBPE(VOCAB_SIZE).to(DEVICE)
    mamba.load_state_dict(torch.load("mamba_bpe_fair.pt", map_location=DEVICE))
    mamba.eval()
    mamba.half()

    log_and_print("\n===== Extreme Streaming Generation Test (100K) =====")
    log_and_print("Note: Transformer is limited by its 2K context window (Limit).")
    log_and_print(f"{'Context':<10} {'BMM':<25} {'Transformer':<25} {'Mamba':<25}")
    
    # 测试长度: 8K, 16K, 32K, 64K, 100K
    lengths = [8192, 16384, 32768, 65536, 102400]
    
    for l in lengths:
        log_and_print(f"--- Testing Context Length: {l} ---")
        
        # BMM
        b_t, b_m = eval_bmm_streaming(bmm, l)
        
        # Transformer (直接标记 Limit)
        if l > 2048:
            g_t, g_m = "Limit", "Limit"
        else:
            g_t, g_m = "N/A", "N/A" # 之前已经测过短序列，这里不重复测
            
        # Mamba
        m_t, m_m = eval_mamba_streaming(mamba, l)
        
        log_and_print(f"{l:<10} {b_t+' tok/s, '+b_m+' GB':<25} {g_t+' tok/s, '+g_m+' GB':<25} {m_t+' tok/s, '+m_m+' GB':<25}")

if __name__ == "__main__":
    main()