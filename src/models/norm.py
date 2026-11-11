import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


def get_norm(norm_type: str, dim: int, eps: float = 1e-8) -> nn.Module:
    key = norm_type.strip().lower()
    if key == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    elif key == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type '{norm_type}'.")
