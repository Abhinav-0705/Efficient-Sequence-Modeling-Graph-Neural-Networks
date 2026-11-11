import math
from typing import Optional

import torch
import torch.nn as nn

from .norm import get_norm
from .attention import MultiHeadAttention
from .mamba import MambaBlock
from .deltanet import DeltaNetLayer_Core


class HybridEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, layer_idx: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int = None, ffn_dim: int = None, dropout: float = 0.1, norm_type: str = "rmsnorm"):
        super().__init__()
        self.is_mamba = (layer_idx % 2 == 0)
        self.norm = get_norm(norm_type, d_model)
        if self.is_mamba:
            self.block = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank, dropout=dropout)
        else:
            self.block = DeltaNetLayer_Core(d_model, num_heads=num_heads, dropout=dropout)
            if ffn_dim is None:
                ffn_dim = d_model * 4
            self.norm2 = get_norm(norm_type, d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(ffn_dim, d_model),
                nn.Dropout(p=dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.block(self.norm(x))
        if not self.is_mamba:
            x = x + self.ffn(self.norm2(x))
        return x


class HybridDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, layer_idx: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int = None, ffn_dim: int = None, dropout: float = 0.1, norm_type: str = "rmsnorm", max_seq_len: int = 512):
        super().__init__()
        self.is_mamba = (layer_idx % 2 == 0)
        self.norm1 = get_norm(norm_type, d_model)
        self.norm_cross = get_norm(norm_type, d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.attn_drop = nn.Dropout(p=dropout)
        if self.is_mamba:
            self.block = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank, dropout=dropout)
        else:
            self.block = DeltaNetLayer_Core(d_model, num_heads=num_heads, dropout=dropout)
            if ffn_dim is None:
                ffn_dim = d_model * 4
            self.norm2 = get_norm(norm_type, d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(ffn_dim, d_model),
                nn.Dropout(p=dropout),
            )

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mem_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.block(self.norm1(x))
        if not self.is_mamba:
            x = x + self.ffn(self.norm2(x))
        x = x + self.attn_drop(self.cross_attn(self.norm_cross(x), memory, memory, mem_mask))
        return x


class HybridSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = None,
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
            HybridEncoderLayer(
                d_model=d_model, num_heads=num_heads, layer_idx=i,
                d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank,
                ffn_dim=ffn_dim, dropout=dropout, norm_type=norm_type,
            )
            for i in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            HybridDecoderLayer(
                d_model=d_model, num_heads=num_heads, layer_idx=i,
                d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank,
                ffn_dim=ffn_dim, dropout=dropout, norm_type=norm_type, max_seq_len=max_seq_len,
            )
            for i in range(num_layers)
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
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

