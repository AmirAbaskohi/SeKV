from __future__ import annotations

import math

import torch
from torch import nn

from sekv.backbone import ModelSpec
from sekv.config import MemoryConfig, RoutingConfig


class RankGatePredictor(nn.Module):
    """Shared g_phi for per-(span, kv_head) rank gates."""

    def __init__(self, r_max: int, head_dim: int, hidden: int = 64):
        super().__init__()
        if r_max <= 0:
            raise ValueError(f"r_max must be positive, got {r_max}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")

        self.r_max = int(r_max)
        self.head_dim = int(head_dim)

        in_dim = self.r_max + self.head_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, self.r_max, bias=True),
        )

    def forward(
        self,
        singular_values: torch.Tensor,
        context: torch.Tensor,
        valid_rank: torch.Tensor,
    ) -> torch.Tensor:
        """Predict masked soft gates m in (0,1) for each singular component."""
        if singular_values.ndim != 2 or singular_values.size(1) != self.r_max:
            raise ValueError(
                "singular_values must have shape [N, r_max], got "
                f"{tuple(singular_values.shape)}"
            )
        if context.ndim != 2 or context.size(0) != singular_values.size(0):
            raise ValueError(
                f"context must have shape [N, d_h], got {tuple(context.shape)}"
            )
        if context.size(1) != self.head_dim:
            raise ValueError(
                f"context second dim must be head_dim={self.head_dim}, got {context.size(1)}"
            )
        if valid_rank.ndim != 1 or valid_rank.size(0) != singular_values.size(0):
            raise ValueError(
                f"valid_rank must have shape [N], got {tuple(valid_rank.shape)}"
            )

        x_sv = singular_values.to(dtype=torch.float32)
        x_ctx = context.to(dtype=torch.float32)
        rank = valid_rank.to(device=singular_values.device, dtype=torch.long)

        eps = 1.0e-8
        denom = x_sv[:, :1] + eps
        normalized_sv = x_sv / denom

        mlp_in = torch.cat([normalized_sv, x_ctx], dim=-1)
        logits = self.mlp(mlp_in)
        gates = torch.sigmoid(logits)

        idx = torch.arange(self.r_max, device=gates.device).unsqueeze(0)
        mask = idx < rank.unsqueeze(1)
        return gates * mask.to(dtype=gates.dtype)


class SeKVModules(nn.Module):
    """All trainable parameters. Base LLM stays frozen elsewhere."""

    def __init__(self, spec: ModelSpec, routing: RoutingConfig, memory: MemoryConfig):
        super().__init__()

        if memory.summary_dim <= 0:
            raise ValueError(f"memory.summary_dim must be positive, got {memory.summary_dim}")
        if memory.r_max <= 0:
            raise ValueError(f"memory.r_max must be positive, got {memory.r_max}")

        self.spec = spec
        self.summary_dim = int(memory.summary_dim)

        self.W = nn.Parameter(
            torch.empty(
                spec.num_layers,
                spec.num_q_heads,
                self.summary_dim,
                spec.head_dim,
                dtype=torch.float32,
            )
        )
        with torch.no_grad():
            for layer in range(spec.num_layers):
                for head in range(spec.num_q_heads):
                    nn.init.xavier_uniform_(self.W[layer, head])

        self.tau = nn.Parameter(
            torch.full(
                (spec.num_layers, spec.num_q_heads),
                float(routing.tau_init),
                dtype=torch.float32,
            )
        )

        self.g_phi = RankGatePredictor(r_max=memory.r_max, head_dim=spec.head_dim)

    def routing_summaries(self, layer: int, k_bar: torch.Tensor) -> torch.Tensor:
        """Compute per-query-head normalized routing summaries from per-kv-head means."""
        if layer < 0 or layer >= self.spec.num_layers:
            raise IndexError(f"layer out of range: {layer}")
        if k_bar.ndim != 3:
            raise ValueError(f"k_bar must have shape [num_spans, H_kv, d_h], got {tuple(k_bar.shape)}")
        if k_bar.size(1) != self.spec.num_kv_heads or k_bar.size(2) != self.spec.head_dim:
            raise ValueError(
                "k_bar shape mismatch for model spec: expected "
                f"[* , {self.spec.num_kv_heads}, {self.spec.head_dim}], got {tuple(k_bar.shape)}"
            )

        num_spans = k_bar.size(0)
        device = k_bar.device

        kv_index = torch.arange(self.spec.num_q_heads, device=device, dtype=torch.long)
        kv_index = kv_index // self.spec.group_size

        k_for_q = k_bar.to(dtype=torch.float32).index_select(dim=1, index=kv_index)
        w_layer = self.W[layer].to(device=device, dtype=torch.float32)

        k_proj = torch.einsum("hpd,nhd->nhp", w_layer, k_for_q)
        norm = torch.linalg.norm(k_proj, dim=-1, keepdim=True).clamp_min(1.0e-8)
        out = k_proj / norm

        if out.shape != (num_spans, self.spec.num_q_heads, self.summary_dim):
            raise RuntimeError(f"Unexpected routing summary shape: {tuple(out.shape)}")
        return out

    def project_query(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        """Project decode-step per-query-head vectors with W[layer] (no normalization)."""
        if layer < 0 or layer >= self.spec.num_layers:
            raise IndexError(f"layer out of range: {layer}")
        if q.ndim != 2 or q.size(0) != self.spec.num_q_heads or q.size(1) != self.spec.head_dim:
            raise ValueError(
                f"q must have shape [{self.spec.num_q_heads}, {self.spec.head_dim}], "
                f"got {tuple(q.shape)}"
            )

        w_layer = self.W[layer].to(device=q.device, dtype=torch.float32)
        q_fp32 = q.to(dtype=torch.float32)
        return torch.einsum("hpd,hd->hp", w_layer, q_fp32)

    def relevance_gate(
        self,
        q_proj: torch.Tensor,
        K_bar: torch.Tensor,
        span_len: torch.Tensor,
    ) -> torch.Tensor:
        """Eq.5 routing gate with per-span log-length prior."""
        if q_proj.ndim != 2 or q_proj.size(0) != self.spec.num_q_heads or q_proj.size(1) != self.summary_dim:
            raise ValueError(
                f"q_proj must have shape [{self.spec.num_q_heads}, {self.summary_dim}], "
                f"got {tuple(q_proj.shape)}"
            )
        if K_bar.ndim != 3 or K_bar.size(1) != self.spec.num_q_heads or K_bar.size(2) != self.summary_dim:
            raise ValueError(
                "K_bar must have shape [num_spans, H, d_prime], got "
                f"{tuple(K_bar.shape)}"
            )
        if span_len.ndim != 1 or span_len.size(0) != K_bar.size(0):
            raise ValueError(
                f"span_len must have shape [num_spans], got {tuple(span_len.shape)}"
            )
        if torch.any(span_len <= 0):
            raise ValueError("span_len must be strictly positive for log(span_len)")

        q_proj_fp32 = q_proj.to(dtype=torch.float32)
        k_bar_fp32 = K_bar.to(device=q_proj.device, dtype=torch.float32)
        span_len_fp32 = span_len.to(device=q_proj.device, dtype=torch.float32)

        logits = torch.einsum("hd,nhd->nh", q_proj_fp32, k_bar_fp32)
        logits = logits / math.sqrt(float(self.summary_dim))
        logits = logits + torch.log(span_len_fp32).unsqueeze(1)
        return torch.sigmoid(logits)

    def trainable_parameters(self) -> dict[str, torch.Tensor]:
        named: dict[str, torch.Tensor] = {
            "W": self.W,
            "tau": self.tau,
        }
        for name, param in self.g_phi.named_parameters():
            named[f"g_phi.{name}"] = param
        return named
