from __future__ import annotations

from typing import Callable

import torch

# positions [n] -> (cos, sin), each [n, d_h] in fp32.
RopeFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x[..., n, d_h] (or x[..., d_h] with n=1) by cos/sin of shape [n, d_h]."""
    x32 = x.to(dtype=torch.float32)
    cos = cos.to(device=x32.device, dtype=torch.float32)
    sin = sin.to(device=x32.device, dtype=torch.float32)
    return x32 * cos + rotate_half(x32) * sin


def invert_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact inverse of apply_rope.

    (cos I + sin R)^-1 = (cos I - sin R) / (cos^2 + sin^2) because R^2 = -I and paired dims
    share cos/sin. The denominator is 1 for plain RoPE and absorbs YaRN attention scaling.
    """
    x32 = x.to(dtype=torch.float32)
    cos = cos.to(device=x32.device, dtype=torch.float32)
    sin = sin.to(device=x32.device, dtype=torch.float32)
    denom = (cos * cos + sin * sin).clamp_min(1.0e-12)
    return (x32 * cos - rotate_half(x32) * sin) / denom
