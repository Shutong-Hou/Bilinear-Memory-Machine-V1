# eval_needle_lambada.py
import os, time, math, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from bmm_model import BMM, prepare_data_bpe, get_batch
from transformers import GPT2Config, GPT2LMHeadModel
from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams
from datasets import load_dataset
import tiktoken

# ==================== 配置 ====================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 50257
LOG_FILE = "needle_lambada_results.txt"

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

# ==================== 模型定义 (与之前完全一致) ====================
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

# ==================== 大海捞针评估 ====================
def eval_needle_in_haystack(model, model_name, context_len, num_samples=10):
    """评估大海捞针任务"""
    log_and_print(f"--- Evaluating {model_name} on Needle In A Haystack (Len={context_len}) ---")
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    # 准备草堆 (使用 WikiText 验证集)
    _, val_data, _, _ = prepare_data_bpe()
    
    # 针和问题
    needle_text = " The magic number for the vault is 8392."
    question_text = " What is the magic number for the vault?"
    answer_text = " 8392"
    
    needle_ids = enc.encode(needle_text)
    question_ids = enc.encode(question_text)
    answer_ids = enc.encode(answer_text)
    
    correct = 0
    
    for _ in range(num_samples):
        # 随机选择插入位置 (0% 到 90%)
        depth = random.uniform(0, 0.9)
        insert_pos = int(depth * (context_len - len(needle_ids) - len(question_ids) - 1))
        
        # 构建输入序列
        start_idx = random.randint(0, len(val_data) - context_len - 10)
        hay = val_data[start_idx : start_idx + insert_pos].tolist()
        hay.extend(needle_ids)
        hay.extend(val_data[start_idx + insert_pos : start_idx + context_len - len(needle_ids) - len(question_ids)].tolist())
        hay.extend(question_ids)
        
        input_ids = torch.tensor([hay], device=DEVICE)
        
        try:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                if model_name == 'Transformer':
                    if input_ids.shape[1] > model.block_size:
                        return "OOM"
                    logits = model(input_ids)
                    pred_id = logits[:, -1, :].argmax(dim=-1).item()
                elif model_name == 'Mamba':
                    inference_params = InferenceParams(max_seqlen=input_ids.shape[1], max_batch_size=1)
                    logits = model(input_ids, inference_params=inference_params)
                    pred_id = logits[:, -1, :].argmax(dim=-1).item()
                else: # BMM
                    logits = model(input_ids)
                    pred_id = logits[:, -1, :].argmax(dim=-1).item()
                    
            if pred_id in answer_ids:
                correct += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return "OOM"
            else:
                raise e
                
    acc = correct / num_samples
    log_and_print(f"  Accuracy: {acc:.2f}")
    return acc

# ==================== LAMBADA 评估 ====================
def eval_lambada(model, model_name):
    """评估 LAMBADA 任务"""
    log_and_print(f"--- Evaluating {model_name} on LAMBADA ---")
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    try:
        dataset = load_dataset("lambada", split="validation")
    except:
        log_and_print("  Failed to load LAMBADA dataset. Skipping.")
        return "N/A"
        
    correct = 0
    total = 0
    
    for item in dataset:
        text = item['text']
        # 分割文本和目标词
        words = text.split()
        if len(words) < 2: continue
        target_word = words[-1]
        context_text = " ".join(words[:-1])
        
        context_ids = enc.encode(context_text)
        target_ids = enc.encode(" " + target_word) # 注意前导空格
        
        if len(target_ids) == 0: continue
        
        input_ids = torch.tensor([context_ids], device=DEVICE)
        
        try:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
                if model_name == 'Transformer':
                    if input_ids.shape[1] > model.block_size:
                        input_ids = input_ids[:, -model.block_size:]
                    logits = model(input_ids)
                    pred_id = logits[:, -1, :].argmax(dim=-1).item()
                elif model_name == 'Mamba':
                    inference_params = InferenceParams(max_seqlen=input_ids.shape[1], max_batch_size=1)
                    logits = model(input_ids, inference_params=inference_params)
                    pred_id = logits[:, -1, :].argmax(dim=-1).item()
                else:
                    logits = model(input_ids)
                    pred_id = logits[:, -1, :].argmax(dim=-1).item()
                    
            if pred_id == target_ids[0]:
                correct += 1
            total += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                
    acc = correct / total if total > 0 else 0
    log_and_print(f"  Accuracy: {acc:.4f}")
    return acc

def main():
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    
    log_and_print("Loading models...")
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

    # ==================== 1. Needle In A Haystack ====================
    log_and_print("\n===== Needle In A Haystack Test =====")
    lengths = [1024, 2048, 4096, 8192]
    results = {'BMM': {}, 'Transformer': {}, 'Mamba': {}}
    
    for l in lengths:
        results['BMM'][l] = eval_needle_in_haystack(bmm, 'BMM', l)
        results['Transformer'][l] = eval_needle_in_haystack(gpt, 'Transformer', l)
        results['Mamba'][l] = eval_needle_in_haystack(mamba, 'Mamba', l)
        
    log_and_print("\n===== Needle In A Haystack Summary =====")
    log_and_print(f"{'Length':<10} {'BMM':<15} {'Transformer':<15} {'Mamba':<15}")
    for l in lengths:
        b = results['BMM'].get(l, 'N/A')
        t = results['Transformer'].get(l, 'N/A')
        m = results['Mamba'].get(l, 'N/A')
        b_str = f"{b:.2f}" if isinstance(b, float) else b
        t_str = f"{t:.2f}" if isinstance(t, float) else t
        m_str = f"{m:.2f}" if isinstance(m, float) else m
        log_and_print(f"{l:<10} {b_str:<15} {t_str:<15} {m_str:<15}")

    # ==================== 2. LAMBADA ====================
    log_and_print("\n===== LAMBADA Test =====")
    lambada_results = {
        'BMM': eval_lambada(bmm, 'BMM'),
        'Transformer': eval_lambada(gpt, 'Transformer'),
        'Mamba': eval_lambada(mamba, 'Mamba')
    }
    log_and_print("\n===== LAMBADA Summary =====")
    for name, acc in lambada_results.items():
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else acc
        log_and_print(f"{name}: {acc_str}")

if __name__ == "__main__":
    main()