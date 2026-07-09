# bmm_model.py
# 统一模型定义文件 - 包含所有架构修复与 BPE 数据加载
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
from datasets import load_dataset

# ==================== 模型架构定义 ====================
class LearnableScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) * 0.1)
    def forward(self, x):
        return x * torch.clamp(self.weight, -2.0, 2.0)

class BilinearMemoryBlockV3(nn.Module):
    def __init__(self, dim, state_dim, memory_slots=128, rank=32,
                 use_bilinear=True, use_memory=True, use_time=True,
                 use_memory_update=False, memory_inject_scale=0.0): # 默认禁用动态更新
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
            # Memory 现在是纯粹的静态可学习参数
            self.memory_init = nn.Parameter(torch.randn(memory_slots, dim) * 0.02)
            
        if use_time:
            self.time_alpha = nn.Parameter(torch.tensor(0.3))

        self.W_g = nn.Parameter(torch.randn(dim, state_dim) * 0.02)
        self.norm_x = LearnableScale(dim)
        self.norm_state = LearnableScale(state_dim)

    def forward(self, x, state, memory=None):
        B, T, D = x.shape
        S, N = self.state_dim, self.memory_slots

        # 修复1：时序循环边界补0
        if self.use_time:
            x_time = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
            x = x + self.time_alpha * x_time

        xf = x.reshape(B * T, D)

        inter_s = 0.0
        if self.use_bilinear:
            x_proj = xf @ self.L_x
            state_flat = state.reshape(B * T, S)
            # 修复2：利用结合律改变Einsum顺序，避免生成(B*T, S, D)的巨大中间张量
            temp = torch.einsum('bs,rsd->brd', state_flat, self.R_xs)
            inter_s = torch.einsum('br,brd->bd', x_proj, temp)
            inter_s = torch.clamp(inter_s, -5.0, 5.0)

        mem_ctx = 0.0
        if self.use_memory:
            gate_read = torch.tanh(xf @ self.W_read)
            # 直接使用静态 memory_init，梯度正常回传
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
        # 返回 None 以保持接口兼容
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
        # 修复3：位置编码循环复用，支持超长序列外推
        pos_idx = torch.arange(T, device=idx.device) % self.pos_phase.shape[1]
        x = self.embed(idx) + self.pos_phase[:, pos_idx, :]
        
        state = torch.zeros(B, T, self.blocks[0].state_dim, device=x.device)
        memory = None
        
        for blk in self.blocks:
            x, state, memory = blk(x, state, memory)
        return self.head(x)

# ==================== BPE 数据加载工具 ====================
def prepare_data_bpe():
    """
    使用 GPT-2 的 BPE Tokenizer (vocab_size=50257) 加载 WikiText-103。
    这是绝对严谨的子词级实验设置。
    """
    print("Loading WikiText-103 and encoding with BPE (GPT-2)...")
    enc = tiktoken.get_encoding("gpt2")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # 将训练集文本拼接，并用 BPE 编码
    texts = dataset["train"]["text"]
    raw_text = "\n".join([t for t in texts if t.strip() != ""])
    data = enc.encode_ordinary(raw_text)
    data = torch.tensor(data, dtype=torch.long)
    
    # 90% 训练，10% 验证
    n = int(len(data) * 0.9)
    return data[:n], data[n:], None, None

def get_batch(data, bs, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (bs,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

# 兼容旧脚本调用 (如果有脚本调用 prepare_data，它会自动指向 BPE)
def prepare_data(max_chars=None):
    return prepare_data_bpe()