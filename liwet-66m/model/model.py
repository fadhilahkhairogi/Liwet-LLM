"""Mini GPT-style decoder-only Transformer, written from scratch in PyTorch.

Architecture: pre-norm + RMSNorm, rotary positional embeddings (RoPE),
causal self-attention via scaled_dot_product_attention, and SwiGLU MLP.
Default config is ~50.4M parameters with a 65-token character vocabulary.
"""

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, base: float, device, dtype):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim); cos/sin: (block_size, head_dim)
    t = x.size(-2)
    cos = cos[:t].unsqueeze(0).unsqueeze(0)
    sin = sin[:t].unsqueeze(0).unsqueeze(0)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x, cos, sin):
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden = config.mlp_hidden
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        cos, sin = build_rope_cache(
            config.block_size,
            config.n_embd // config.n_head,
            config.rope_base,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._rope_device = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope(self, device, dtype):
        if self._rope_device != (device, dtype):
            self.rope_cos = self.rope_cos.to(device=device, dtype=dtype)
            self.rope_sin = self.rope_sin.to(device=device, dtype=dtype)
            self._rope_device = (device, dtype)
        return self.rope_cos, self.rope_sin

    def forward(self, idx, targets=None):
        b, t = idx.shape
        if t > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            if targets is not None:
                targets = targets[:, -self.config.block_size:]
            t = idx.size(1)

        x = self.drop(self.tok_emb(idx))
        cos, sin = self._get_rope(idx.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=200,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
    ):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    scores = logits[b, seen]
                    logits[b, seen] = torch.where(
                        scores > 0, scores / repetition_penalty, scores * repetition_penalty
                    )

            logits = logits / max(temperature, 1e-8)

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < thresh, float("-inf"))

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = probs.cumsum(dim=-1)
                remove = cumulative - probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in params.values() if p.dim() >= 2]
        nodecay_params = [p for p in params.values() if p.dim() < 2]
        groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=fused)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.config.tie_weights:
                n -= self.lm_head.weight.numel()
        return n


def build_model(config: GPTConfig) -> MiniTransformer:
    return MiniTransformer(config)


if __name__ == "__main__":
    cfg = GPTConfig()
    model = build_model(cfg)
    total = model.num_parameters()
    print(f"config            : {cfg.to_dict()}")
    print(f"total parameters  : {total:,} (~{total / 1e6:.2f}M)")
    print(f"non-embedding     : {model.num_parameters(non_embedding=True):,}")
