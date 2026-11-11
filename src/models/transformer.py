import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .positional import SinusoidalPositionalEncoding
from .norm import get_norm


class FFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1, norm_type="rmsnorm", max_seq_len=512):
        super().__init__()
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.ffn = FFN(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, normed, normed, mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.1, norm_type="rmsnorm", max_seq_len=512):
        super().__init__()
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)
        self.norm3 = get_norm(norm_type, d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.ffn = FFN(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, memory, tgt_mask=None, mem_mask=None):
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, tgt_mask))
        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, memory, memory, mem_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        norm_type: str = "rmsnorm",
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pe = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.tgt_pe = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ffn_dim, dropout, norm_type, max_seq_len)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, ffn_dim, dropout, norm_type, max_seq_len)
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

    def encode(self, src, src_mask=None):
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.src_pe(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)

    def decode(self, tgt, memory, tgt_mask=None, mem_mask=None):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pe(x)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, mem_mask)
        return self.decoder_norm(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, mem_mask=None):
        memory = self.encode(src, src_mask)
        x = self.decode(tgt, memory, tgt_mask, mem_mask)
        return self.output_proj(x)

