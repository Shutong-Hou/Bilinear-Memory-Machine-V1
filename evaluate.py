# final_objective_eval.py
import os
# 必须在导入其他库之前强制设置镜像，防止网络报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bmm_model import BMM, prepare_data_bpe, get_batch
from transformers import GPT2Config, GPT2LMHeadModel
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 50257
LOG_FILE = "final_objective_results.txt"

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

# ==================== 真实模型定义 (完全复现你的 bmm_model.py) ====================
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

# ==================== 基线模型定义 (严格对齐训练脚本) ====================
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

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.embed_dim, self.n_head = embed_dim, n_head
        self.head_dim = embed_dim // n_head
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, causal_mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None: att = att + causal_mask
        att = F.softmax(att, dim=-1)
        att = (att @ v).transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(att)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_mult=4):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * ff_mult)
        self.fc2 = nn.Linear(embed_dim * ff_mult, embed_dim)
        self.act = nn.GELU()
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_head, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_head)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_mult)
    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, block_size=2048, n_layer=24, n_head=12, n_embd=768, ff_mult=4):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, ff_mult) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.head.weight = self.token_embed.weight
        if self.head.bias is not None:
            self.head.bias.requires_grad = False
            self.head.bias.data.zero_()
    def forward(self, idx):
        B, T = idx.shape
        if T > self.block_size: idx = idx[:, -self.block_size:]; T = self.block_size
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        mask = torch.triu(torch.ones(T, T, device=idx.device) * float('-inf'), diagonal=1)
        for blk in self.blocks: x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# ==================== 客观评估函数 ====================
def eval_ppl_rigorous(model, val_data, seq_len, model_type='bmm'):
    model.eval()
    batch_size = 8 if seq_len <= 1024 else (4 if seq_len <= 2048 else 1)
    total_loss = 0.0
    total_tokens = 0
    num_samples = 5
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
        for _ in range(num_samples):
            xv, yv = get_batch(val_data, batch_size, seq_len, DEVICE)
            if model_type == 'gpt' and seq_len > model.block_size:
                return "Limit"
            logits = model(xv)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yv.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += xv.numel()
    return math.exp(total_loss / total_tokens)

def eval_bmm_throughput(model, context_len, gen_len=50):
    """BMM 逐 token 生成效率 (O(1) 显存测试)"""
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (1, context_len), device=DEVICE)
    max_pos = model.pos_phase.shape[1]
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
            # 逐 token 处理 prompt，确保显存不随 T 增长
            current_token = prompt[:, 0:1]
            x = model.embed(current_token) + model.pos_phase[:, 0:1, :]
            state = torch.zeros(1, 1, model.blocks[0].state_dim, device=DEVICE)
            for blk in model.blocks: x, state, _ = blk(x, state)
            for i in range(1, context_len):
                current_token = prompt[:, i:i+1]
                pos_idx = i % max_pos
                x = model.embed(current_token) + model.pos_phase[:, pos_idx:pos_idx+1, :]
                for blk in model.blocks: x, state, _ = blk(x, state)
            next_token = model.head(x).argmax(dim=-1)
            current_pos = context_len
            
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

def eval_gpt_throughput(context_len, gen_len=50):
    config = GPT2Config(vocab_size=VOCAB_SIZE, n_positions=2048, n_embd=768, n_layer=24, n_head=12)
    hf_gpt = GPT2LMHeadModel(config).to(DEVICE).eval()
    if context_len >= 2048: return "Limit", "Limit"
    input_ids = torch.randint(0, VOCAB_SIZE, (1, context_len), device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = hf_gpt(input_ids, use_cache=True)
        past_kv = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(gen_len):
            out = hf_gpt(next_token, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        t1 = time.time()
    throughput = gen_len / (t1 - t0)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    return f"{throughput:.1f}", f"{peak_mem:.2f}"

def eval_mamba_throughput(model, context_len, gen_len=50):
    model.eval()
    if context_len > 16384: return "Limit", "Limit"
    prompt = torch.randint(0, VOCAB_SIZE, (1, context_len), device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            inference_params = InferenceParams(max_seqlen=context_len + gen_len, max_batch_size=1)
            logits = model(prompt, inference_params=inference_params)
            next_token = logits[:, -1, :].argmax(dim=-1)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(gen_len):
                logits = model(next_token.unsqueeze(1), inference_params=inference_params)
                next_token = logits[:, -1, :].argmax(dim=-1)
            torch.cuda.synchronize()
            t1 = time.time()
    except Exception:
        return "OOM", "OOM"
    throughput = gen_len / (t1 - t0)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    return f"{throughput:.1f}", f"{peak_mem:.2f}

def main():
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    log_and_print("Preparing BPE data...")
    _, val_data, _, _ = prepare_data_bpe()

    log_and_print("\nLoading models...")
    bmm = BMM_Tied(vocab_size=VOCAB_SIZE, embed_dim=768, state_dim=768, num_blocks=8,
                   memory_slots=128, rank=32, max_pos=2048,
                   use_bilinear=True, use_memory=True, use_time=True, 
                   use_memory_update=False, memory_inject_scale=2.0).to(DEVICE)
    bmm.load_state_dict(torch.load("bmm_bpe_gamma2.0.pt", map_location=DEVICE))
    bmm.eval()

    gpt = GPT(VOCAB_SIZE, block_size=2048, n_layer=24, n_head=12, n_embd=768).to(DEVICE)
    gpt.load_state_dict(torch.load("gpt_bpe_truly_fair.pt", map_location=DEVICE))
    gpt.eval()

    mamba = MambaBPE(VOCAB_SIZE).to(DEVICE)
    mamba.load_state_dict(torch.load("mamba_bpe_fair.pt", map_location=DEVICE))
    mamba.eval()
    mamba.half()

    # ==================== Table 1: Main PPL ====================
    log_and_print("\n===== Table 1: Main Validation PPL (seq_len=512) =====")
    bmm_ppl = eval_ppl_rigorous(bmm, val_data, 512, model_type='bmm')
    gpt_ppl = eval_ppl_rigorous(gpt, val_data, 512, model_type='gpt')
    mamba_ppl = eval_ppl_rigorous(mamba, val_data, 512, model_type='mamba')
    log_and_print(f"BMM (50k, γ=2.0): {bmm_ppl:.2f}")
    log_and_print(f"Transformer (50k): {gpt_ppl:.2f}")
    log_and_print(f"Mamba (50k): {mamba_ppl:.2f}")

    # ==================== Table 2: Long Sequence PPL ====================
    log_and_print("\n===== Table 2: Long Sequence PPL =====")
    lengths = [512, 1024, 2048, 4096, 8192]
    log_and_print(f"{'Length':<8} {'BMM':<10} {'Transformer':<15} {'Mamba':<10}")
    for l in lengths:
        b_ppl = eval_ppl_rigorous(bmm, val_data, l, model_type='bmm')
        g_ppl = eval_ppl_rigorous(gpt, val_data, l, model_type='gpt')
        m_ppl = eval_ppl_rigorous(mamba, val_data, l, model_type='mamba')
        b_str = f"{b_ppl:.2f}" if isinstance(b_ppl, float) else b_ppl
        g_str = f"{g_ppl:.2f}" if isinstance(g_ppl, float) else g_ppl
        m_str = f"{m_ppl:.2f}" if isinstance(m_ppl, float) else m_ppl
        log_and_print(f"{l:<8} {b_str:<10} {g_str:<15} {m_str:<10}")

    # ==================== Table 3: Inference Efficiency ====================
    log_and_print("\n===== Table 3: Generation Throughput (tok/s) & Memory (GB) =====")
    log_and_print("Note: Transformer uses HF KV Cache, Mamba uses Native InferenceParams (FP16), BMM uses O(1) State.")
    log_and_print(f"{'Context':<10} {'BMM':<20} {'Transformer':<20} {'Mamba':<20}")
    for l in [128, 512, 1024, 2048, 4096, 8192]:
        b_t, b_m = eval_bmm_throughput(bmm, l)
        g_t, g_m = eval_gpt_throughput(l)
        m_t, m_m = eval_mamba_throughput(mamba, l)
        log_and_print(f"{l:<10} {b_t+' tok/s, '+b_m+' GB':<20} {g_t+' tok/s, '+g_m+' GB':<20} {m_t+' tok/s, '+m_m+' GB':<20}")

if __name__ == "__main__":
    main()