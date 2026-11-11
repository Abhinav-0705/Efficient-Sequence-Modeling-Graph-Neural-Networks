import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import get_norm
from .attention import MultiHeadAttention


class DeltaNetLayer_Core(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        Q = self.q_proj(x).view(B, T, H, D)
        K = self.k_proj(x).view(B, T, H, D)
        V = self.v_proj(x).view(B, T, H, D)
        beta = torch.sigmoid(self.beta_proj(x))

        K = F.normalize(K, p=2, dim=-1)
        Q = F.normalize(Q, p=2, dim=-1)

        outputs = []
        S = torch.zeros(B, H, D, D, device=x.device, dtype=x.dtype)

        for t in range(T):
            k_t = K[:, t, :, :]
            v_t = V[:, t, :, :]
            q_t = Q[:, t, :, :]
            b_t = beta[:, t, :]

            Sk = torch.matmul(S, k_t.unsqueeze(-1)).squeeze(-1)
            erase = torch.matmul(Sk.unsqueeze(-1), k_t.unsqueeze(-2))
            write = torch.matmul(v_t.unsqueeze(-1), k_t.unsqueeze(-2))
            S = S - b_t.unsqueeze(-1).unsqueeze(-1) * erase + write
            o_t = torch.matmul(S, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(o_t)

        out = torch.stack(outputs, dim=1)
        out = out.view(B, T, self.d_model)
        return self.dropout(self.o_proj(out))


class DeltaNetBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, ffn_dim: int = None, dropout: float = 0.1, norm_type: str = "rmsnorm"):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = d_model * 4
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)
        self.delta = DeltaNetLayer_Core(d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.delta(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DeltaNetDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int = None, dropout: float = 0.1, norm_type: str = "rmsnorm", max_seq_len: int = 512):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = d_model * 4
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)
        self.norm3 = get_norm(norm_type, d_model)
        self.delta = DeltaNetLayer_Core(d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(p=dropout),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mem_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.delta(self.norm1(x))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, mem_mask))
        x = x + self.ffn(self.norm3(x))
        return x


class DeltaNetSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        norm_type: str = "rmsnorm",
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        if ffn_dim is None:
            ffn_dim = d_model * 4
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.encoder_layers = nn.ModuleList([
            DeltaNetBlock(d_model, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout, norm_type=norm_type)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DeltaNetDecoderBlock(d_model, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout, norm_type=norm_type, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.encoder_norm = get_norm(norm_type, d_model)
        self.decoder_norm = get_norm(norm_type, d_model)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.d_model ** -0.5)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.src_embed(src) * math.sqrt(self.d_model)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.encoder_norm(x)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, mem_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        for layer in self.decoder_layers:
            x = layer(x, memory, mem_mask)
        return self.decoder_norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, mem_mask=None):
        memory = self.encode(src)
        x = self.decode(tgt, memory, mem_mask)
        return self.output_proj(x)

