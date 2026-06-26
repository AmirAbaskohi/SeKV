from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from sekv.backbone import FrozenBackbone, PrefillOutputs
from sekv.config import SeKVConfig
from sekv.segmentation import SegmentationResult


@dataclass
class TeacherSignals:
    query_positions: list[int]
    span_mass: dict[int, torch.Tensor]
    zoom_targets: dict[int, torch.Tensor]
    teacher_logits: dict[int, torch.Tensor]


@contextmanager
def _force_eager_attention(backbone: FrozenBackbone):
    model = backbone.model
    cfg = model.config

    old_public = getattr(cfg, "attn_implementation", None)
    old_private = getattr(cfg, "_attn_implementation", None)

    if hasattr(cfg, "attn_implementation"):
        setattr(cfg, "attn_implementation", "eager")
    if hasattr(cfg, "_attn_implementation"):
        setattr(cfg, "_attn_implementation", "eager")

    try:
        yield
    finally:
        if hasattr(cfg, "attn_implementation"):
            setattr(cfg, "attn_implementation", old_public)
        if hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", old_private)


def _run_teacher_forward(
    backbone: FrozenBackbone,
    input_ids: torch.Tensor,
    output_attentions: bool,
    output_hidden_states: bool = False,
):
    ids = input_ids
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.size(0) != 1:
        raise ValueError(f"input_ids must be [seq_len] or [1, seq_len], got {tuple(input_ids.shape)}")

    ids = ids.to(device=backbone.model.device)

    with torch.no_grad(), _force_eager_attention(backbone):
        out = backbone.model(
            input_ids=ids,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
    return out


def extract_teacher_attention(
    backbone: FrozenBackbone,
    input_ids: torch.Tensor,
    query_positions: list[int],
) -> torch.Tensor:
    """Extract [Q,L,H,seq_len] attention rows for requested query positions in fp32 on CPU."""
    if not query_positions:
        num_layers = backbone.spec.num_layers
        num_heads = backbone.spec.num_q_heads
        seq_len = int(input_ids.numel()) if input_ids.ndim == 1 else int(input_ids.size(-1))
        return torch.empty((0, num_layers, num_heads, seq_len), dtype=torch.float32, device="cpu")

    seq_len = int(input_ids.numel()) if input_ids.ndim == 1 else int(input_ids.size(-1))
    for pos in query_positions:
        if pos < 0 or pos >= seq_len:
            raise ValueError(f"query position out of bounds: {pos} for seq_len={seq_len}")

    out = _run_teacher_forward(backbone, input_ids, output_attentions=True)
    if out.attentions is None:
        raise RuntimeError("Teacher forward did not return attentions")

    layers_rows: list[torch.Tensor] = []
    q_idx = torch.tensor(query_positions, dtype=torch.long, device=out.attentions[0].device)

    for layer_attn in out.attentions:
        # layer_attn: [B,H,Q,K]
        rows = layer_attn[0].index_select(dim=1, index=q_idx).to(dtype=torch.float32)
        # rows: [H,Q,K] -> [Q,H,K]
        layers_rows.append(rows.permute(1, 0, 2).contiguous())

    stacked = torch.stack(layers_rows, dim=1)  # [Q,L,H,K]
    return stacked.detach().to(device="cpu", dtype=torch.float32).contiguous()


def aggregate_span_mass(attn_rows: torch.Tensor, seg: SegmentationResult) -> torch.Tensor:
    """Aggregate teacher attention mass over span tokens: [Q,L,H,num_spans]."""
    if attn_rows.ndim != 4:
        raise ValueError(f"attn_rows must be [Q,L,H,seq_len], got {tuple(attn_rows.shape)}")
    if attn_rows.size(-1) != seg.seq_len:
        raise ValueError(
            f"attn_rows seq_len mismatch: got {attn_rows.size(-1)}, expected {seg.seq_len}"
        )

    q, l, h, _ = attn_rows.shape
    num_spans = len(seg.spans)
    out = torch.zeros((q, l, h, num_spans), dtype=torch.float32, device=attn_rows.device)

    for span_idx, span in enumerate(seg.spans):
        if not span.token_indices:
            continue
        idx = torch.tensor(span.token_indices, dtype=torch.long, device=attn_rows.device)
        out[..., span_idx] = attn_rows.index_select(dim=-1, index=idx).sum(dim=-1)

    return out.to(dtype=torch.float32)


def coverage_targets(span_mass: torch.Tensor, rho: float) -> torch.Tensor:
    """Build binary coverage targets per (q,l,h) over spans only."""
    if span_mass.ndim != 4:
        raise ValueError(f"span_mass must be [Q,L,H,S], got {tuple(span_mass.shape)}")
    if rho <= 0.0 or rho > 1.0:
        raise ValueError(f"rho must be in (0,1], got {rho}")

    q, l, h, s = span_mass.shape
    if s == 0:
        return torch.zeros_like(span_mass, dtype=torch.float32)

    mass = span_mass.to(dtype=torch.float32)
    total = mass.sum(dim=-1, keepdim=True)
    norm_mass = torch.where(total > 0.0, mass / total.clamp_min(1.0e-12), torch.zeros_like(mass))

    y = torch.zeros_like(norm_mass, dtype=torch.float32)

    flat_mass = norm_mass.view(-1, s)
    flat_y = y.view(-1, s)

    for i in range(flat_mass.size(0)):
        row = flat_mass[i]
        if float(row.sum().item()) <= 0.0:
            continue
        order = torch.argsort(row, descending=True)
        sorted_row = row.index_select(0, order)
        csum = torch.cumsum(sorted_row, dim=0)
        k = int(torch.searchsorted(csum, torch.tensor(float(rho), device=csum.device), right=False).item())
        k = min(k, s - 1)
        keep = order[: k + 1]
        flat_y[i, keep] = 1.0

    return y


def _mine_query_positions_with_candidates(
    attn_full: torch.Tensor,
    candidate_positions: list[int],
    window: int,
    max_positions: int,
) -> list[int]:
    if attn_full.ndim != 4:
        raise ValueError(f"attn_full must be [Q,L,H,seq_len], got {tuple(attn_full.shape)}")
    if len(candidate_positions) != attn_full.size(0):
        raise ValueError(
            "candidate_positions length must match first attn_full dimension: "
            f"{len(candidate_positions)} vs {attn_full.size(0)}"
        )
    if max_positions <= 0:
        return []

    seq_len = int(attn_full.size(-1))
    scores: list[tuple[float, int]] = []

    for i, qpos in enumerate(candidate_positions):
        if qpos <= 0:
            continue
        left_end = max(0, qpos - window)
        if left_end == 0:
            frac = 0.0
        else:
            far_mass = attn_full[i, :, :, :left_end].sum()
            total_mass = attn_full[i, :, :, :qpos].sum().clamp_min(1.0e-12)
            frac = float((far_mass / total_mass).item())
        scores.append((frac, qpos))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [pos for _, pos in scores[:max_positions]]


def mine_query_positions(
    attn_full: torch.Tensor,
    seg: SegmentationResult,
    window: int,
    max_positions: int,
) -> list[int]:
    """Mine high long-range-dependency query rows from provided attention tensor rows."""
    del seg
    candidate_positions = list(range(attn_full.size(0)))
    return _mine_query_positions_with_candidates(attn_full, candidate_positions, window, max_positions)


def build_teacher_signals(
    backbone: FrozenBackbone,
    input_ids: torch.Tensor,
    prefill: PrefillOutputs,
    seg: SegmentationResult,
    cfg: SeKVConfig,
) -> TeacherSignals:
    """Mine positions, aggregate teacher span mass, create rho-coverage labels and logits."""
    del prefill

    ids = input_ids
    if ids.ndim == 2:
        if ids.size(0) != 1:
            raise ValueError(f"input_ids batch must be 1, got {tuple(ids.shape)}")
        ids = ids[0]
    if ids.ndim != 1:
        raise ValueError(f"input_ids must be [seq_len] or [1, seq_len], got {tuple(input_ids.shape)}")

    seq_len = int(ids.numel())
    if seq_len <= 1:
        return TeacherSignals(query_positions=[], span_mass={}, zoom_targets={}, teacher_logits={})

    candidate_count = min(max(8, cfg.train.mining_window), seq_len - 1)
    candidate_positions = torch.linspace(1, seq_len - 1, steps=candidate_count, dtype=torch.float32)
    candidate_positions = candidate_positions.round().to(dtype=torch.long).unique(sorted=True).tolist()

    out = _run_teacher_forward(backbone, ids.unsqueeze(0), output_attentions=True)
    if out.attentions is None:
        raise RuntimeError("Teacher forward did not return attentions")

    q_idx = torch.tensor(candidate_positions, dtype=torch.long, device=out.attentions[0].device)
    layers_rows: list[torch.Tensor] = []
    for layer_attn in out.attentions:
        rows = layer_attn[0].index_select(dim=1, index=q_idx).to(dtype=torch.float32)
        layers_rows.append(rows.permute(1, 0, 2).contiguous())
    attn_candidates = torch.stack(layers_rows, dim=1).detach().to(device="cpu", dtype=torch.float32)

    max_positions = min(64, len(candidate_positions))
    mined_positions = _mine_query_positions_with_candidates(
        attn_full=attn_candidates,
        candidate_positions=candidate_positions,
        window=cfg.train.mining_window,
        max_positions=max_positions,
    )

    if not mined_positions:
        mined_positions = candidate_positions[: min(8, len(candidate_positions))]

    mined_idx = [candidate_positions.index(p) for p in mined_positions]
    attn_rows = attn_candidates.index_select(
        dim=0,
        index=torch.tensor(mined_idx, dtype=torch.long, device=attn_candidates.device),
    )

    span_mass_all = aggregate_span_mass(attn_rows, seg)
    zoom_target_all = coverage_targets(span_mass_all, rho=cfg.loss.rho)

    teacher_logits_dict: dict[int, torch.Tensor] = {}
    for pos in mined_positions:
        teacher_logits_dict[int(pos)] = out.logits[0, pos].detach().to(device="cpu", dtype=torch.float32)

    span_mass_dict: dict[int, torch.Tensor] = {}
    zoom_targets_dict: dict[int, torch.Tensor] = {}
    for i, pos in enumerate(mined_positions):
        span_mass_dict[int(pos)] = span_mass_all[i].detach().to(device="cpu", dtype=torch.float32)
        zoom_targets_dict[int(pos)] = zoom_target_all[i].detach().to(device="cpu", dtype=torch.float32)

    return TeacherSignals(
        query_positions=[int(p) for p in mined_positions],
        span_mass=span_mass_dict,
        zoom_targets=zoom_targets_dict,
        teacher_logits=teacher_logits_dict,
    )
