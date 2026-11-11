import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import get_norm
from .attention import MultiHeadAttention


class SelectiveSSM(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = None, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        if dt_rank is None:
            dt_rank = max(1, d_inner // 16)
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        log_A = torch.log(torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1))
        self.A_log = nn.Parameter(log_A)
        self.D = nn.Parameter(torch.ones(d_inner))

        nn.init.uniform_(self.dt_proj.bias, math.log(dt_min), math.log(dt_max))
        self.dt_proj.bias.data = torch.exp(self.dt_proj.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        d_state = self.d_state

        A = -torch.exp(self.A_log.float())

        x_proj = self.x_proj(x)
        dt_raw = x_proj[..., :self.dt_rank]
        B_proj = x_proj[..., self.dt_rank: self.dt_rank + d_state]
        C_proj = x_proj[..., self.dt_rank + d_state:]

        dt = F.softplus(self.dt_proj(dt_raw))

        h = torch.zeros(B, d, d_state, device=x.device, dtype=x.dtype)
        ys = []
        A_unsqueezed = A.unsqueeze(0)
        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)
            deltaA_t = torch.exp(dt_t * A_unsqueezed)
            deltaB_t = dt_t * B_proj[:, t, :].unsqueeze(1) * x[:, t, :].unsqueeze(-1)
            h = deltaA_t * h + deltaB_t
            y_t = (h * C_proj[:, t, :].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)

        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        d_inner = self.d_inner

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner, bias=True)
        self.ssm = SelectiveSSM(d_inner, d_state=d_state, dt_rank=dt_rank)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


class MambaEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int = None, dropout: float = 0.1, norm_type: str = "rmsnorm"):
        super().__init__()
        self.norm = get_norm(norm_type, d_model)
        self.block = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(self.norm(x))


class MambaDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: int = None, dropout: float = 0.1, norm_type: str = "rmsnorm", max_seq_len: int = 512):
        super().__init__()
        self.norm1 = get_norm(norm_type, d_model)
        self.norm2 = get_norm(norm_type, d_model)
        self.block = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mem_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.block(self.norm1(x))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, mem_mask))
        return x


class MambaSeq2Seq(nn.Module):
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
        self.encoder_layers = nn.ModuleList([
            MambaEncoderLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank, dropout=dropout, norm_type=norm_type)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            MambaDecoderLayer(d_model, num_heads, d_state=d_state, d_conv=d_conv, expand=expand, dt_rank=dt_rank, dropout=dropout, norm_type=norm_type, max_seq_len=max_seq_len)
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

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None, tgt_mask=None, mem_mask=None) -> torch.Tensor:
        memory = self.encode(src)
        x = self.decode(tgt, memory, mem_mask)
        return self.output_proj(x)

