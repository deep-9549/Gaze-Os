"""Small causal transformer language model (L2, PRD 14.1): 4 layers, 256 dims.

Word-level vocabulary (30k): the runtime scores whole next words with one
softmax, which keeps CPU latency under 30 ms. See docs/adr/0003 for why this
replaces the byte-pair encoding mentioned in the PRD. Blocks are written out
explicitly so the ONNX export keeps a dynamic sequence length.
"""

from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        h = self.heads
        q = q.reshape(B, T, h, C // h).transpose(1, 2)
        k = k.reshape(B, T, h, C // h).transpose(1, 2)
        v = v.reshape(B, T, h, C // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (C // h) ** -0.5
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.mlp(self.ln2(x)))


class SmallGPT(nn.Module):
    def __init__(self, vocab: int, dim: int = 256, layers: int = 4, heads: int = 4, context: int = 32,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab, bias=False)
        self.out.weight = self.tok.weight  # weight tying keeps it near 10M parameters
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        # GPT-style small init: with tied weights, the default N(0, 1) embedding
        # init makes the output logits huge and training diverge.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.shape[1]
        pos = torch.arange(T, device=tokens.device)
        x = self.tok(tokens) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.out(self.norm(x))


