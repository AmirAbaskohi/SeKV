from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from sekv.backbone import ModelSpec
from sekv.config import MemoryConfig, SeKVConfig
from sekv.memory import LayerMemory, reconstruct_span
from sekv.modules import SeKVModules


@dataclass(frozen=True)
class GroupEntries:
    K: torch.Tensor
    V: torch.Tensor
    score_bias: torch.Tensor
    entry_kind: torch.Tensor
    span_of_entry: torch.Tensor
    coarse_row_of_span: dict[int, int]
    fine_rows_of_span: dict[int, list[int]]


def ste_zoom_gate(alpha_tilde: torch.Tensor, tau: torch.Tensor, temperature: float) -> torch.Tensor:
    """Straight-through zoom gate with hard forward and soft backward."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    alpha = alpha_tilde.to(dtype=torch.float32)
    tau_tensor = tau.to(device=alpha.device, dtype=torch.float32)

    z_hard = (alpha > tau_tensor).to(dtype=torch.float32)
    surrogate = torch.sigmoid((alpha - tau_tensor) / float(temperature))
    return z_hard + (surrogate - surrogate.detach())


def masked_softmax_attention(
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    score_bias: torch.Tensor,
    log_gate: torch.Tensor,
) -> torch.Tensor:
    """Single-softmax attention over mixed-resolution entries for one query head."""
    if q.ndim != 1:
        raise ValueError(f"q must have shape [d_h], got {tuple(q.shape)}")
    if K.ndim != 2 or V.ndim != 2:
        raise ValueError(f"K and V must have shape [M, d_h], got K{tuple(K.shape)} V{tuple(V.shape)}")
    if K.shape != V.shape:
        raise ValueError(f"K and V shapes must match, got {tuple(K.shape)} and {tuple(V.shape)}")

    m = K.size(0)
    d_h = K.size(1)

    if score_bias.ndim != 1 or score_bias.numel() != m:
        raise ValueError(f"score_bias must have shape [M], got {tuple(score_bias.shape)}")
    if log_gate.ndim != 1 or log_gate.numel() != m:
        raise ValueError(f"log_gate must have shape [M], got {tuple(log_gate.shape)}")

    if m == 0:
        return torch.zeros(d_h, device=q.device, dtype=torch.float32)

    q_f32 = q.to(dtype=torch.float32)
    k_f32 = K.to(device=q.device, dtype=torch.float32)
    v_f32 = V.to(device=q.device, dtype=torch.float32)

    bias = score_bias.to(device=q.device, dtype=torch.float32)
    lg = log_gate.to(device=q.device, dtype=torch.float32)

    scores = torch.mv(k_f32, q_f32) / math.sqrt(float(d_h))
    scores = scores + bias + lg

    row_max = torch.max(scores)
    probs_unnorm = torch.exp(scores - row_max)
    probs = probs_unnorm / probs_unnorm.sum().clamp_min(1.0e-12)
    return torch.matmul(probs, v_f32)


def budget_select(
    alpha_tilde_head: torch.Tensor,
    tau_head: torch.Tensor,
    span_lengths: torch.Tensor,
    layer_mem: LayerMemory,
    budget: int,
) -> torch.Tensor:
    """Per-head span expansion under threshold-first then alpha-sorted budget pruning."""
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")

    alpha = alpha_tilde_head.to(dtype=torch.float32)
    tau = tau_head.to(device=alpha.device, dtype=torch.float32)
    lengths = span_lengths.to(device=alpha.device, dtype=torch.long)

    if alpha.ndim != 1 or lengths.ndim != 1 or alpha.numel() != lengths.numel():
        raise ValueError(
            "alpha_tilde_head and span_lengths must be 1D tensors with identical length"
        )

    num_spans = int(alpha.numel())
    default_expand = alpha > tau

    num_anchors = int(layer_mem.anchor_k.size(1))
    num_full_tokens = 0
    for span in layer_mem.spans:
        if span.full_resolution:
            num_full_tokens += int(span.length)

    # Baseline persistent residency: anchors + one coarse row per non-full span + full-res tokens.
    persistent = num_anchors + num_spans + num_full_tokens
    if persistent > budget:
        raise ValueError(
            f"Persistent KV residency ({persistent}) exceeds budget ({budget}) before any expansion"
        )

    extra_if_expanded = (lengths - 1).clamp_min(0)
    proposed_total = persistent + int(extra_if_expanded[default_expand].sum().item())
    if proposed_total <= budget:
        return default_expand

    expand_mask = torch.zeros_like(default_expand, dtype=torch.bool)
    candidate_indices = torch.nonzero(default_expand, as_tuple=False).squeeze(-1)
    if candidate_indices.numel() == 0:
        return expand_mask

    candidate_scores = alpha.index_select(0, candidate_indices)
    order = torch.argsort(candidate_scores, descending=True)

    used = persistent
    for ordered_idx in order.tolist():
        span_idx = int(candidate_indices[ordered_idx].item())
        extra = int(extra_if_expanded[span_idx].item())
        if used + extra <= budget:
            expand_mask[span_idx] = True
            used += extra

    if used > budget:
        raise RuntimeError("Budget selection produced a plan that exceeds budget")

    return expand_mask


def assemble_group_entries(
    layer_mem: LayerMemory,
    group: int,
    modules: SeKVModules,
    layer: int,
    spans_to_reconstruct: set[int],
    memory_cfg: MemoryConfig,
) -> GroupEntries:
    """Build the KV-group entry universe: anchors + coarse + full-res + optional recon fine rows."""
    if group < 0 or group >= modules.spec.num_kv_heads:
        raise IndexError(f"group index out of range: {group}")

    if memory_cfg.r_max <= 0:
        raise ValueError(f"memory_cfg.r_max must be > 0, got {memory_cfg.r_max}")

    device = layer_mem.anchor_k.device
    d_h = modules.spec.head_dim

    k_rows: list[torch.Tensor] = []
    v_rows: list[torch.Tensor] = []
    score_bias_rows: list[float] = []
    entry_kind_rows: list[int] = []
    span_of_entry_rows: list[int] = []

    coarse_row_of_span: dict[int, int] = {}
    fine_rows_of_span: dict[int, list[int]] = {}

    # Anchors are full-resolution entries (always-gated-on).
    if layer_mem.anchor_k.ndim != 3 or layer_mem.anchor_v.ndim != 3:
        raise ValueError("layer_mem.anchor_k/anchor_v must be rank-3 [H_kv, N, d_h]")
    if layer_mem.anchor_k.shape != layer_mem.anchor_v.shape:
        raise ValueError("layer_mem.anchor_k and anchor_v shape mismatch")

    num_anchors = int(layer_mem.anchor_k.size(1))
    if num_anchors > 0:
        anchor_k_group = layer_mem.anchor_k[group].to(device=device, dtype=torch.float32)
        anchor_v_group = layer_mem.anchor_v[group].to(device=device, dtype=torch.float32)

        k_rows.append(anchor_k_group)
        v_rows.append(anchor_v_group)
        score_bias_rows.extend([0.0] * num_anchors)
        entry_kind_rows.extend([0] * num_anchors)
        span_of_entry_rows.extend([-1] * num_anchors)

    for span_id, span in enumerate(layer_mem.spans):
        if span.full_resolution:
            if span.raw_k is None or span.raw_v is None:
                raise ValueError(f"Span {span_id} is full_resolution but raw_k/raw_v are missing")

            raw_k_group = span.raw_k[group].to(device=device, dtype=torch.float32)
            raw_v_group = span.raw_v[group].to(device=device, dtype=torch.float32)
            num_tokens = int(raw_k_group.size(0))

            if num_tokens != span.length:
                raise ValueError(
                    f"Span {span_id} length mismatch: expected {span.length}, got {num_tokens}"
                )

            if num_tokens > 0:
                k_rows.append(raw_k_group)
                v_rows.append(raw_v_group)
                score_bias_rows.extend([0.0] * num_tokens)
                entry_kind_rows.extend([0] * num_tokens)
                span_of_entry_rows.extend([span_id] * num_tokens)
            continue

        # Non-full span: always include one coarse row with log|S| bias.
        coarse_row_idx = len(score_bias_rows)
        coarse_row_of_span[span_id] = coarse_row_idx

        k_rows.append(span.k_bar[group].to(device=device, dtype=torch.float32).unsqueeze(0))
        v_rows.append(span.v_bar[group].to(device=device, dtype=torch.float32).unsqueeze(0))
        score_bias_rows.append(math.log(float(span.length)))
        entry_kind_rows.append(1)
        span_of_entry_rows.append(span_id)

        if span_id not in spans_to_reconstruct:
            continue

        if span.factors is None:
            raise ValueError(f"Span {span_id} selected for reconstruction but factors are missing")

        factors = span.factors
        rank = int(factors.valid_rank)
        if rank <= 0:
            raise ValueError(f"Span {span_id} has non-positive valid_rank: {rank}")

        r_max = int(memory_cfg.r_max)
        singular_padded = torch.zeros((1, r_max), dtype=torch.float32, device=device)
        singular_src = factors.s_k[group].to(device=device, dtype=torch.float32)
        if singular_src.numel() != rank:
            raise ValueError(
                f"Span {span_id} singular value length mismatch: expected {rank}, got {singular_src.numel()}"
            )
        singular_padded[:, :rank] = singular_src

        context = span.k_bar[group].to(device=device, dtype=torch.float32)
        context = context / context.norm(p=2).clamp_min(1.0e-8)
        context = context.unsqueeze(0)

        valid_rank = torch.tensor([rank], dtype=torch.long, device=device)
        gates_full = modules.g_phi(singular_padded, context, valid_rank)
        gates = gates_full[:, :rank]

        U_k = factors.U_k[group].to(device=device, dtype=torch.float32).unsqueeze(0)
        s_k = factors.s_k[group].to(device=device, dtype=torch.float32).unsqueeze(0)
        Vmat_k = factors.Vmat_k[group].to(device=device, dtype=torch.float32).unsqueeze(0)

        U_v = factors.U_v[group].to(device=device, dtype=torch.float32).unsqueeze(0)
        s_v = factors.s_v[group].to(device=device, dtype=torch.float32).unsqueeze(0)
        Vmat_v = factors.Vmat_v[group].to(device=device, dtype=torch.float32).unsqueeze(0)

        fine_k = reconstruct_span(U_k, s_k, Vmat_k, gates).squeeze(0)
        fine_v = reconstruct_span(U_v, s_v, Vmat_v, gates).squeeze(0)

        if fine_k.size(0) != span.length or fine_v.size(0) != span.length:
            raise ValueError(
                f"Span {span_id} reconstruction length mismatch: expected {span.length}, "
                f"got K={fine_k.size(0)} V={fine_v.size(0)}"
            )

        fine_start = len(score_bias_rows)
        fine_end = fine_start + span.length
        fine_rows_of_span[span_id] = list(range(fine_start, fine_end))

        k_rows.append(fine_k)
        v_rows.append(fine_v)
        score_bias_rows.extend([0.0] * span.length)
        entry_kind_rows.extend([2] * span.length)
        span_of_entry_rows.extend([span_id] * span.length)

    if not k_rows:
        empty = torch.empty((0, d_h), dtype=torch.float32, device=device)
        empty_i64 = torch.empty((0,), dtype=torch.long, device=device)
        return GroupEntries(
            K=empty,
            V=empty,
            score_bias=torch.empty((0,), dtype=torch.float32, device=device),
            entry_kind=empty_i64,
            span_of_entry=empty_i64,
            coarse_row_of_span=coarse_row_of_span,
            fine_rows_of_span=fine_rows_of_span,
        )

    K = torch.cat(k_rows, dim=0)
    V = torch.cat(v_rows, dim=0)
    score_bias = torch.tensor(score_bias_rows, dtype=torch.float32, device=device)
    entry_kind = torch.tensor(entry_kind_rows, dtype=torch.long, device=device)
    span_of_entry = torch.tensor(span_of_entry_rows, dtype=torch.long, device=device)

    return GroupEntries(
        K=K,
        V=V,
        score_bias=score_bias,
        entry_kind=entry_kind,
        span_of_entry=span_of_entry,
        coarse_row_of_span=coarse_row_of_span,
        fine_rows_of_span=fine_rows_of_span,
    )


def build_log_gate(
    group_entries: GroupEntries,
    z_per_span: dict[int, torch.Tensor],
    head_in_group: int,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Build per-head additive log-gate mask over one group's entry universe."""
    if eps <= 0.0 or eps >= 1.0:
        raise ValueError(f"eps must be in (0,1), got {eps}")

    m = int(group_entries.score_bias.numel())
    log_gate = torch.zeros((m,), dtype=torch.float32, device=group_entries.score_bias.device)

    for span_id, z_heads in z_per_span.items():
        if head_in_group < 0 or head_in_group >= int(z_heads.numel()):
            raise IndexError(
                f"head_in_group={head_in_group} out of range for z_per_span[{span_id}]"
            )

        z = z_heads[head_in_group].to(device=log_gate.device, dtype=torch.float32)

        coarse_row = group_entries.coarse_row_of_span.get(span_id)
        if coarse_row is not None:
            coarse_gate = torch.clamp(1.0 - z, min=eps, max=1.0)
            log_gate[coarse_row] = torch.log(coarse_gate)

        fine_rows = group_entries.fine_rows_of_span.get(span_id)
        if fine_rows:
            fine_gate = torch.clamp(z, min=eps, max=1.0)
            fine_log = torch.log(fine_gate)
            row_idx = torch.tensor(fine_rows, dtype=torch.long, device=log_gate.device)
            log_gate.index_fill_(0, row_idx, fine_log)

    return log_gate


def decode_step_layer(
    q_layer: torch.Tensor,
    layer_mem: LayerMemory,
    modules: SeKVModules,
    layer: int,
    spec: ModelSpec,
    cfg: SeKVConfig,
    budget: int | None,
    training: bool,
    ste_temperature: float,
) -> torch.Tensor:
    """Run one decode step for a layer with per-head zoom routing and mixed single-softmax."""
    if q_layer.ndim != 2 or q_layer.size(0) != spec.num_q_heads or q_layer.size(1) != spec.head_dim:
        raise ValueError(
            f"q_layer must have shape [{spec.num_q_heads}, {spec.head_dim}], got {tuple(q_layer.shape)}"
        )
    if layer < 0 or layer >= spec.num_layers:
        raise IndexError(f"layer out of range: {layer}")

    device = q_layer.device
    output = torch.zeros((spec.num_q_heads, spec.head_dim), dtype=torch.float32, device=device)

    compressed_span_ids = [i for i, span in enumerate(layer_mem.spans) if not span.full_resolution]

    if compressed_span_ids:
        k_bar = torch.stack([layer_mem.spans[i].k_bar for i in compressed_span_ids], dim=0)
        k_bar = k_bar.to(device=device, dtype=torch.float32)

        span_len = torch.tensor(
            [layer_mem.spans[i].length for i in compressed_span_ids],
            dtype=torch.float32,
            device=device,
        )

        K_bar = modules.routing_summaries(layer, k_bar)
        q_proj = modules.project_query(layer, q_layer)
        alpha_tilde = modules.relevance_gate(q_proj, K_bar, span_len)

        tau_layer = modules.tau[layer].to(device=device, dtype=torch.float32)
        z = ste_zoom_gate(alpha_tilde, tau_layer, ste_temperature)
        z_hard = (alpha_tilde > tau_layer).to(dtype=torch.bool)
    else:
        alpha_tilde = torch.empty((0, spec.num_q_heads), dtype=torch.float32, device=device)
        z = torch.empty((0, spec.num_q_heads), dtype=torch.float32, device=device)
        z_hard = torch.empty((0, spec.num_q_heads), dtype=torch.bool, device=device)
        span_len = torch.empty((0,), dtype=torch.float32, device=device)

    for group in range(spec.num_kv_heads):
        head_start = group * spec.group_size
        head_end = head_start + spec.group_size
        head_slice = slice(head_start, head_end)

        # Default per-head gate values used in log-gate assembly.
        z_group_effective: dict[int, torch.Tensor] = {}

        spans_to_reconstruct: set[int] = set()
        if compressed_span_ids:
            if training:
                for local_idx, span_id in enumerate(compressed_span_ids):
                    if torch.any(z_hard[local_idx, head_slice]):
                        spans_to_reconstruct.add(span_id)
                    z_group_effective[span_id] = z[local_idx, head_slice]
            else:
                if budget is None:
                    for local_idx, span_id in enumerate(compressed_span_ids):
                        if torch.any(z_hard[local_idx, head_slice]):
                            spans_to_reconstruct.add(span_id)
                        z_group_effective[span_id] = z_hard[local_idx, head_slice].to(dtype=torch.float32)
                else:
                    per_head_select: list[torch.Tensor] = []
                    for h in range(head_start, head_end):
                        select_h = budget_select(
                            alpha_tilde_head=alpha_tilde[:, h],
                            tau_head=tau_layer[h],
                            span_lengths=span_len,
                            layer_mem=layer_mem,
                            budget=budget,
                        )
                        per_head_select.append(select_h)

                    select_matrix = torch.stack(per_head_select, dim=1)
                    for local_idx, span_id in enumerate(compressed_span_ids):
                        if torch.any(select_matrix[local_idx]):
                            spans_to_reconstruct.add(span_id)
                        z_group_effective[span_id] = select_matrix[local_idx].to(
                            device=device, dtype=torch.float32
                        )

        group_entries = assemble_group_entries(
            layer_mem=layer_mem,
            group=group,
            modules=modules,
            layer=layer,
            spans_to_reconstruct=spans_to_reconstruct,
            memory_cfg=cfg.memory,
        )

        for offset, head in enumerate(range(head_start, head_end)):
            q_h = q_layer[head].to(device=device, dtype=torch.float32)
            log_gate = build_log_gate(group_entries, z_group_effective, offset)

            output[head] = masked_softmax_attention(
                q=q_h,
                K=group_entries.K,
                V=group_entries.V,
                score_bias=group_entries.score_bias,
                log_gate=log_gate,
            )

    return output
