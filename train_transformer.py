# train_transformer_bpe_truly_fair.py
import os, time, math
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
# 修改：直接复用 BMM 的数据加载逻辑，确保数据完全一致
from bmm_model import prepare_data_bpe, get_batch

VOCAB_SIZE = 50257
SEQ_LEN = 512
BATCH_SIZE = 8
STEPS = 50000
WARMUP = 2000
LR = 3e-4
WEIGHT_DECAY = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # 修改：结构维度与 BMM 对齐 (n_embd=768, n_head=12)
    # 24层参数量约208M，与BMM的197M非常接近
    def __init__(self, vocab_size, block_size=2048, n_layer=24, n_head=12, n_embd=768, ff_mult=4):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, ff_mult) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        # 公平起见，Transformer 也使用 Weight Tying
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

def main():
    print("Preparing BPE data (using unified loader)...")
    train_data, val_data, _, _ = prepare_data_bpe()
    
    model = GPT(VOCAB_SIZE, block_size=2048).to(DEVICE)
    print(f"GPT BPE (Truly Fair) parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()

        if step % 1000 == 0 or step == 1:
            model.eval()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                xv, yv = get_batch(val_data, BATCH_SIZE*2, SEQ_LEN, DEVICE)
                val_loss = F.cross_entropy(model(xv).view(-1, VOCAB_SIZE), yv.view(-1))
                ppl = math.exp(min(val_loss.item(), 20))
                current_lr = optimizer.param_groups[0]['lr']
                print(f"GPT Step {step:5d} | Val PPL: {ppl:.2f} | LR: {current_lr:.2e}")

    total_time = time.time() - t_start
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
        xv, yv = get_batch(val_data, BATCH_SIZE*4, SEQ_LEN, DEVICE)
        val_loss = F.cross_entropy(model(xv).view(-1, VOCAB_SIZE), yv.view(-1))
        final_ppl = math.exp(val_loss.item())
    print(f"GPT BPE (Truly Fair) Final PPL: {final_ppl:.2f}, Time: {total_time:.1f}s")
    torch.save(model.state_dict(), "gpt_bpe_truly_fair.pt")

if __name__ == "__main__":
    main()