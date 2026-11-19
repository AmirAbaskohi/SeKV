from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from sekv.backbone import ModelSpec
from sekv.config import MemoryConfig, RoutingConfig

KIND_KEY = 0
KIND_VALUE = 1


class RankGatePredictor(nn.Module):
    """Shared g_phi: gates m_i in (0,1) per singular component from the spectrum and span summary.

    Keys and values are gated separately; a one-hot kind flag tells the predictor which one it
    is scoring (values are typically harder to compress and receive more components).
    """

    def __init__(self, r_max: int, head_dim: int, hidden: int = 64, init_gate_logit: float = 3.0):
        super().__init__()
        if r_max <= 0:
            raise ValueError(f"r_max must be positive, got {r_max}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")

        self.r_max = int(r_max)
        self.head_dim = int(head_dim)

        in_dim = self.r_max + self.head_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=True),
            nn.GELU(),
            nn.Linear(hidden, self.r_max, bias=True),
        )
        # Start with nearly open gates so early reconstructions are close to the truncated SVD.
        with torch.no_grad():
            self.mlp[-1].bias.fill_(float(init_gate_logit))

    def forward(
        self,
        singular_values: torch.Tensor,
        context: torch.Tensor,
        valid_rank: torch.Tensor,
        kind: torch.Tensor | int = KIND_KEY,
    ) -> torch.Tensor:
        """singular_values [N, r_max], context [N, d_h], valid_rank [N] -> gates [N, r_max]."""
        if singular_values.ndim != 2 or singular_values.size(1) != self.r_max:
            raise ValueError(
                f"singular_values must have shape [N, {self.r_max}], got {tuple(singular_values.shape)}"
            )
        n = singular_values.size(0)
        if context.ndim != 2 or context.shape != (n, self.head_dim):
            raise ValueError(f"context must have shape [{n}, {self.head_dim}], got {tuple(context.shape)}")
        if valid_rank.ndim != 1 or valid_rank.size(0) != n:
            raise ValueError(f"valid_rank must have shape [{n}], got {tuple(valid_rank.shape)}")

        device = singular_values.device
        x_sv = singular_values.to(dtype=torch.float32)
        normalized_sv = x_sv / (x_sv[:, :1].abs() + 1.0e-8)
        x_ctx = context.to(device=device, dtype=torch.float32)

        if isinstance(kind, int):
            kind_t = torch.full((n,), int(kind), dtype=torch.long, device=device)
        else:
            kind_t = kind.to(device=device, dtype=torch.long).reshape(n)
        kind_onehot = F.one_hot(kind_t, num_classes=2).to(dtype=torch.float32)

        weight_dtype = self.mlp[0].weight.dtype
        logits = self.mlp(torch.cat([normalized_sv, x_ctx, kind_onehot], dim=-1).to(dtype=weight_dtype))
        gates = torch.sigmoid(logits.to(dtype=torch.float32))

        idx = torch.arange(self.r_max, device=device).unsqueeze(0)
        mask = idx < valid_rank.to(device=device, dtype=torch.long).unsqueeze(1)
        return gates * mask.to(dtype=gates.dtype)


class SeKVModules(nn.Module):
    """All trainable parameters (W, tau, g_phi). The base LLM stays frozen elsewhere."""

    def __init__(self, spec: ModelSpec, routing: RoutingConfig, memory: MemoryConfig):
        super().__init__()

        if memory.summary_dim <= 0:
            raise ValueError(f"memory.summary_dim must be positive, got {memory.summary_dim}")
        if memory.r_max <= 0:
            raise ValueError(f"memory.r_max must be positive, got {memory.r_max}")

        self.spec = spec
        self.summary_dim = int(memory.summary_dim)

        # Per-(layer, query head) routing projection W in R^{d' x d_h}.
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

        # Per-(layer, query head) zoom threshold tau.
        self.tau = nn.Parameter(
            torch.full(
                (spec.num_layers, spec.num_q_heads),
                float(routing.tau_init),
                dtype=torch.float32,
            )
        )

        self.g_phi = RankGatePredictor(r_max=memory.r_max, head_dim=spec.head_dim)

    @property
    def r_max(self) -> int:
        return self.g_phi.r_max

    def routing_summaries(self, layer: int, k_bar: torch.Tensor) -> torch.Tensor:
        """Eq.1: K_bar_S = W sum_t w_t K_t / ||.||, i.e. W k_bar_S normalized, per query head.

        k_bar: [num_spans, H_kv, d_h] surprisal-weighted mean keys -> [num_spans, H, d'].
        """
        if layer < 0 or layer >= self.spec.num_layers:
            raise IndexError(f"layer out of range: {layer}")
        if k_bar.ndim != 3 or k_bar.size(1) != self.spec.num_kv_heads or k_bar.size(2) != self.spec.head_dim:
            raise ValueError(
                f"k_bar must have shape [num_spans, {self.spec.num_kv_heads}, {self.spec.head_dim}], "
                f"got {tuple(k_bar.shape)}"
            )

        device = k_bar.device
        kv_index = torch.arange(self.spec.num_q_heads, device=device, dtype=torch.long) // self.spec.group_size
        k_for_q = k_bar.to(dtype=torch.float32).index_select(dim=1, index=kv_index)
        w_layer = self.W[layer].to(device=device, dtype=torch.float32)

        k_proj = torch.einsum("hpd,nhd->nhp", w_layer, k_for_q)
        return k_proj / torch.linalg.norm(k_proj, dim=-1, keepdim=True).clamp_min(1.0e-8)

    def project_query(self, layer: int, q: torch.Tensor) -> torch.Tensor:
        """Project decode-step per-query-head vectors [H, d_h] with W[layer] -> [H, d']."""
        if layer < 0 or layer >= self.spec.num_layers:
            raise IndexError(f"layer out of range: {layer}")
        if q.ndim != 2 or q.size(0) != self.spec.num_q_heads or q.size(1) != self.spec.head_dim:
            raise ValueError(
                f"q must have shape [{self.spec.num_q_heads}, {self.spec.head_dim}], got {tuple(q.shape)}"
            )

        w_layer = self.W[layer].to(device=q.device, dtype=torch.float32)
        return torch.einsum("hpd,hd->hp", w_layer, q.to(dtype=torch.float32))

    def relevance_gate(self, q_proj: torch.Tensor, K_bar: torch.Tensor, span_len: torch.Tensor) -> torch.Tensor:
        """Eq.3: alpha_S = sigmoid((W q)^T K_bar_S / sqrt(d') + log|S|) -> [num_spans, H]."""
        if q_proj.ndim != 2 or q_proj.size(0) != self.spec.num_q_heads or q_proj.size(1) != self.summary_dim:
            raise ValueError(
                f"q_proj must have shape [{self.spec.num_q_heads}, {self.summary_dim}], got {tuple(q_proj.shape)}"
            )
        if K_bar.ndim != 3 or K_bar.size(1) != self.spec.num_q_heads or K_bar.size(2) != self.summary_dim:
            raise ValueError(f"K_bar must have shape [num_spans, H, d_prime], got {tuple(K_bar.shape)}")
        if span_len.ndim != 1 or span_len.size(0) != K_bar.size(0):
            raise ValueError(f"span_len must have shape [num_spans], got {tuple(span_len.shape)}")
        if torch.any(span_len <= 0):
            raise ValueError("span_len must be strictly positive for log(span_len)")

        logits = torch.einsum("hd,nhd->nh", q_proj.to(dtype=torch.float32), K_bar.to(device=q_proj.device, dtype=torch.float32))
        logits = logits / math.sqrt(float(self.summary_dim))
        logits = logits + torch.log(span_len.to(device=q_proj.device, dtype=torch.float32)).unsqueeze(1)
        return torch.sigmoid(logits)

    def rank_gates(
        self,
        singular_values: torch.Tensor,
        k_bar: torch.Tensor,
        valid_rank: torch.Tensor,
        kind: int,
    ) -> torch.Tensor:
        """Gates for stacked factors.

        singular_values [..., R] (R <= r_max, zero padded), k_bar [..., d_h] span summary per KV head,
        valid_rank broadcastable to the leading dims -> gates [..., r_max].
        """
        lead = singular_values.shape[:-1]
        r = singular_values.size(-1)
        if r > self.r_max:
            raise ValueError(f"singular value rank {r} exceeds r_max {self.r_max}")
        sv = F.pad(singular_values.to(dtype=torch.float32), (0, self.r_max - r)).reshape(-1, self.r_max)
        ctx = k_bar.to(dtype=torch.float32)
        ctx = (ctx / ctx.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)).reshape(-1, self.spec.head_dim)
        valid = valid_rank.to(device=sv.device, dtype=torch.long).expand(lead).reshape(-1)
        return self.g_phi(sv, ctx, valid, kind).reshape(*lead, self.r_max)

    def trainable_parameters(self) -> dict[str, torch.Tensor]:
        named: dict[str, torch.Tensor] = {"W": self.W, "tau": self.tau}
        for name, param in self.g_phi.named_parameters():
            named[f"g_phi.{name}"] = param
        return named
