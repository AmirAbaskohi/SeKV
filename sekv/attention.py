from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol

import torch

from sekv.backbone import ModelSpec
from sekv.config import SeKVConfig
from sekv.memory import LayerMemory, derive_layer, reconstruct_span_kv
from sekv.modules import SeKVModules
from sekv.rope import RopeFn

_LOG_EPS = 1.0e-6


class BudgetTooSmallError(ValueError):
    """The minimum resident state (anchors, window, summaries, coarse entries) exceeds the budget."""


class SpanReconstructor(Protocol):
    """Supplies reconstructed (post-RoPE K, V) for a (span, KV group) at decode time."""

    def prepare(self, layer: int, layer_mem: LayerMemory, requests: list[tuple[int, int]]) -> None: ...

    def get(self, layer: int, layer_mem: LayerMemory, span_idx: int, group: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def finish_layer(self, layer: int, layer_mem: LayerMemory, requests: list[tuple[int, int]]) -> None: ...


class DirectReconstructor:
    """Reconstructs spans on demand from their stored factors without caching or transfer tracking."""

    def __init__(self, rope_fn: RopeFn, device: torch.device | str):
        self.rope_fn = rope_fn
        self.device = torch.device(device)

    def prepare(self, layer: int, layer_mem: LayerMemory, requests: list[tuple[int, int]]) -> None:
        del layer, layer_mem, requests

    def get(self, layer: int, layer_mem: LayerMemory, span_idx: int, group: int) -> tuple[torch.Tensor, torch.Tensor]:
        del layer
        span = layer_mem.spans[span_idx]
        cos, sin = self.rope_fn(torch.arange(span.start, span.stop, device=self.device))
        return reconstruct_span_kv(span, cos, sin, group=group, device=self.device)

    def finish_layer(self, layer: int, layer_mem: LayerMemory, requests: list[tuple[int, int]]) -> None:
        del layer, layer_mem, requests


@dataclass
class LayerStepInfo:
    comp_ids: list[int]
    # Routing probabilities alpha_S and zoom decisions z_S for compressed spans, [m, H].
    alpha: torch.Tensor
    z: torch.Tensor
    # Heads that attend to a span at token resolution, [m, H]; spans reconstructed per KV group, [m, H_kv].
    head_fine: torch.Tensor
    expanded: torch.Tensor
    # GPU-resident KV of this layer at this step in token units per KV head.
    resident_tokens: float


def ste_zoom_gate(alpha_tilde: torch.Tensor, tau: torch.Tensor, temperature: float) -> torch.Tensor:
    """Straight-through zoom gate: hard 1[alpha > tau] forward, sigmoid((alpha - tau)/T) backward."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    alpha = alpha_tilde.to(dtype=torch.float32)
    tau_tensor = tau.to(device=alpha.device, dtype=torch.float32)

    z_hard = (alpha > tau_tensor).to(dtype=torch.float32)
    surrogate = torch.sigmoid((alpha - tau_tensor) / float(temperature))
    return z_hard + (surrogate - surrogate.detach())


def summary_cost_tokens(spec: ModelSpec, summary_dim: int) -> float:
    """GPU cost of one span's routing summaries in token-per-KV-head units (H_q d' / 2 H_kv d_h)."""
    return float(spec.num_q_heads * summary_dim) / float(2 * spec.num_kv_heads * spec.head_dim)


def persistent_tokens(layer_mem: LayerMemory, spec: ModelSpec, summary_dim: int, extra_window: int = 0) -> float:
    """Resident KV before any expansion: anchors, exact short spans, window, coarse entries, summaries."""
    derived = derive_layer(layer_mem)
    m = len(derived.comp_ids)
    return float(derived.num_exact + layer_mem.window_len + extra_window) + m * (1.0 + summary_cost_tokens(spec, summary_dim))


def select_under_budget(
    alpha: torch.Tensor,
    hard: torch.Tensor,
    lengths: torch.Tensor,
    spec: ModelSpec,
    capacity_tokens: float,
) -> torch.Tensor:
    """Choose spans to reconstruct per KV group, highest alpha first, until the capacity is filled.

    A (span, group) pair is a candidate if any query head of the group selects the span; its
    score is the largest alpha among the selecting heads and its cost is |S| tokens.
    """
    m = alpha.size(0)
    groups = hard.view(m, spec.num_kv_heads, spec.group_size)
    candidates = groups.any(dim=-1)
    selected = torch.zeros_like(candidates)
    if capacity_tokens <= 0 or not bool(candidates.any()):
        return selected

    scores = torch.where(groups, alpha.view(m, spec.num_kv_heads, spec.group_size), torch.full_like(alpha.view(m, spec.num_kv_heads, spec.group_size), -1.0)).amax(dim=-1)
    cand_idx = candidates.nonzero(as_tuple=False)
    cand_scores = scores[candidates]
    cand_costs = lengths.to(scores.device)[cand_idx[:, 0]]
    order = torch.argsort(cand_scores, descending=True)
    keep = torch.cumsum(cand_costs[order], dim=0) <= float(capacity_tokens) + 1.0e-6
    chosen = cand_idx[order[keep]]
    selected[chosen[:, 0], chosen[:, 1]] = True
    return selected


def decode_step_layer(
    q_layer: torch.Tensor,
    layer_mem: LayerMemory,
    modules: SeKVModules,
    layer: int,
    spec: ModelSpec,
    cfg: SeKVConfig,
    budget: Optional[float] = None,
    training: bool = False,
    ste_temperature: Optional[float] = None,
    reconstructor: Optional[SpanReconstructor] = None,
    summaries: Optional[torch.Tensor] = None,
    train_fine: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    return_info: bool = False,
):
    """One decode step of a layer: per-head routing, budgeted zoom-in, mixed-resolution attention.

    q_layer: post-RoPE queries [H, d_h]. The local window of layer_mem must already contain the
    current token. budget is in tokens per KV head (None = uncapped).

    Training passes `train_fine = (fine_k [H_kv, N, d_h], fine_v, token_to_span [N])` with every
    compressed span reconstructed; non-selected heads are masked out of fine rows and selected
    heads out of coarse rows, with gradients reaching W and tau through the straight-through gate.
    At inference spans are reconstructed per KV group through `reconstructor`.
    """
    if q_layer.ndim != 2 or q_layer.size(0) != spec.num_q_heads or q_layer.size(1) != spec.head_dim:
        raise ValueError(f"q_layer must have shape [{spec.num_q_heads}, {spec.head_dim}], got {tuple(q_layer.shape)}")
    if layer < 0 or layer >= spec.num_layers:
        raise IndexError(f"layer out of range: {layer}")

    device = q_layer.device
    h_kv, g_size, d_h = spec.num_kv_heads, spec.group_size, spec.head_dim
    q = q_layer.to(dtype=torch.float32)
    scale = 1.0 / math.sqrt(float(d_h))
    temperature = float(cfg.routing.ste_temperature if ste_temperature is None else ste_temperature)

    derived = derive_layer(layer_mem)
    comp_ids = derived.comp_ids
    m = len(comp_ids)
    lengths = derived.lengths.to(device)
    persistent = persistent_tokens(layer_mem, spec, modules.summary_dim)

    if m:
        k_bar = derived.k_bar.to(device)
        if summaries is None:
            if training:
                summaries = modules.routing_summaries(layer, k_bar)
            else:
                cached = layer_mem.cache.get("summaries")
                if cached is None or cached.size(0) != m:
                    cached = modules.routing_summaries(layer, k_bar)
                    layer_mem.cache["summaries"] = cached
                summaries = cached
        alpha = modules.relevance_gate(modules.project_query(layer, q), summaries, lengths)
        tau = modules.tau[layer].to(device=device, dtype=torch.float32)
        hard = alpha > tau
        expanded = hard.view(m, h_kv, g_size).any(dim=-1)
        if budget is not None and not training:
            capacity = (float(budget) - persistent) * h_kv
            if capacity < 0:
                raise BudgetTooSmallError(
                    f"Resident state ({persistent:.1f} tokens per KV head) exceeds the budget ({float(budget):.1f})"
                )
            expanded = select_under_budget(alpha, hard, lengths, spec, capacity)
        head_fine = hard & expanded.repeat_interleave(g_size, dim=1)
        z = ste_zoom_gate(alpha, tau, temperature) if training else head_fine.to(torch.float32)
    else:
        k_bar = derived.k_bar.to(device)
        alpha = torch.zeros((0, spec.num_q_heads), dtype=torch.float32, device=device)
        z = torch.zeros_like(alpha)
        head_fine = torch.zeros((0, spec.num_q_heads), dtype=torch.bool, device=device)
        expanded = torch.zeros((0, h_kv), dtype=torch.bool, device=device)

    resident = persistent + float((expanded.to(torch.float32) * lengths.unsqueeze(1)).sum().item()) / h_kv
    if budget is not None and not training and resident > float(budget) + 1.0e-6:
        raise RuntimeError(f"Budget violated at layer {layer}: resident={resident:.1f}, budget={float(budget):.1f}")

    exact_k = derived.exact_k.to(device=device, dtype=torch.float32)
    exact_v = derived.exact_v.to(device=device, dtype=torch.float32)
    win_k = layer_mem.window_k.to(device=device, dtype=torch.float32)
    win_v = layer_mem.window_v.to(device=device, dtype=torch.float32)
    n_static = exact_k.size(1) + win_k.size(1)
    log_len = torch.log(lengths) if m else lengths

    if training and train_fine is not None:
        fine_k, fine_v, token_span = train_fine
        token_span = token_span.to(device)
        K = torch.cat([exact_k, win_k, k_bar.transpose(0, 1), fine_k.to(device=device, dtype=torch.float32)], dim=1)
        V = torch.cat([exact_v, win_v, derived.v_bar.to(device).transpose(0, 1), fine_v.to(device=device, dtype=torch.float32)], dim=1)
        bias = torch.cat([
            torch.zeros(n_static, device=device),
            log_len,
            torch.zeros(token_span.numel(), device=device),
        ])
        z_t = z.transpose(0, 1)
        fine_t = head_fine.transpose(0, 1)
        neg_inf = torch.tensor(float("-inf"), device=device)
        coarse_gate = torch.where(fine_t, neg_inf, torch.log((1.0 - z_t).clamp_min(_LOG_EPS)))
        fine_gate = torch.where(fine_t[:, token_span], torch.log(z_t[:, token_span].clamp_min(_LOG_EPS)), neg_inf)
        gate = torch.cat([torch.zeros((spec.num_q_heads, n_static), device=device), coarse_gate, fine_gate], dim=1)

        scores = torch.einsum("gjd,gmd->gjm", q.view(h_kv, g_size, d_h), K).reshape(spec.num_q_heads, -1) * scale
        probs = torch.softmax(scores + bias.unsqueeze(0) + gate, dim=-1)
        output = torch.einsum("gjm,gmd->gjd", probs.view(h_kv, g_size, -1), V).reshape(spec.num_q_heads, d_h)
    else:
        if reconstructor is None:
            raise ValueError("decode_step_layer needs a reconstructor at inference time")
        requests = [(comp_ids[int(s)], int(g)) for s, g in expanded.nonzero(as_tuple=False).tolist()]
        reconstructor.prepare(layer, layer_mem, requests)

        output = torch.zeros((spec.num_q_heads, d_h), dtype=torch.float32, device=device)
        v_bar = derived.v_bar.to(device)
        for g in range(h_kv):
            heads = slice(g * g_size, (g + 1) * g_size)
            exp_local = expanded[:, g].nonzero(as_tuple=False).squeeze(-1).tolist() if m else []
            k_parts = [exact_k[g], win_k[g], k_bar[:, g]]
            v_parts = [exact_v[g], win_v[g], v_bar[:, g]]
            gate_parts = [
                torch.zeros((g_size, n_static), device=device),
                torch.where(head_fine[:, heads].transpose(0, 1), float("-inf"), 0.0).to(torch.float32),
            ]
            bias_parts = [torch.zeros(n_static, device=device), log_len]
            for s in exp_local:
                fk, fv = reconstructor.get(layer, layer_mem, comp_ids[s], g)
                k_parts.append(fk.to(device=device, dtype=torch.float32))
                v_parts.append(fv.to(device=device, dtype=torch.float32))
                n_tok = int(fk.size(0))
                bias_parts.append(torch.zeros(n_tok, device=device))
                head_mask = torch.where(head_fine[s, heads], 0.0, float("-inf")).to(device=device, dtype=torch.float32)
                gate_parts.append(head_mask.unsqueeze(1).expand(g_size, n_tok))

            K = torch.cat(k_parts, dim=0)
            V = torch.cat(v_parts, dim=0)
            scores = (q[heads] @ K.transpose(0, 1)) * scale + torch.cat(bias_parts).unsqueeze(0) + torch.cat(gate_parts, dim=1)
            output[heads] = torch.softmax(scores, dim=-1) @ V

        reconstructor.finish_layer(layer, layer_mem, requests)

    if not return_info:
        return output
    info = LayerStepInfo(
        comp_ids=list(comp_ids),
        alpha=alpha,
        z=z,
        head_fine=head_fine,
        expanded=expanded,
        resident_tokens=resident,
    )
    return output, info
