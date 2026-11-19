from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from sekv.backbone import FrozenBackbone, ModelSpec, PrefillOutputs
from sekv.config import SeKVConfig
from sekv.rope import RopeFn, apply_rope
from sekv.segmentation import SegmentationResult


@dataclass
class TeacherSignals:
    query_positions: list[int]
    # Full-KV teacher attention mass per span, [L, H, num_spans] for each selected position.
    span_mass: dict[int, torch.Tensor]
    # Share of teacher attention outside the local window, averaged over layers and heads.
    far_fraction: dict[int, float]
    teacher_logits: dict[int, torch.Tensor]


def candidate_query_positions(seq_len: int, cfg: SeKVConfig) -> list[int]:
    """Evenly spaced positions late enough that some context falls outside the local window."""
    lo = int(cfg.memory.local_window) + int(cfg.segmentation.l_gap)
    hi = seq_len - 1
    if hi < lo:
        return []
    count = max(1, min(int(cfg.train.query_candidates), hi - lo + 1))
    positions = torch.linspace(lo, hi, steps=count).round().to(dtype=torch.long).unique(sorted=True)
    return [int(p) for p in positions.tolist()]


def token_span_index(seg: SegmentationResult) -> torch.Tensor:
    """Map each token to its span index; anchors map to the extra bucket len(seg.spans)."""
    index = torch.full((seg.seq_len,), len(seg.spans), dtype=torch.long)
    for span_idx, span in enumerate(seg.spans):
        index[span.start : span.stop] = span_idx
    return index


def teacher_attention_rows(
    queries: torch.Tensor,
    keys: torch.Tensor,
    positions: list[int],
    spec: ModelSpec,
    rope_fn: RopeFn,
) -> torch.Tensor:
    """Causal full-KV attention rows for one layer.

    queries: pre-RoPE [C, H, d_h] at `positions`; keys: post-RoPE [H_kv, n, d_h] -> [C, H, n] fp32.
    """
    device = keys.device
    pos = torch.tensor(positions, dtype=torch.long, device=device)
    cos, sin = rope_fn(pos)
    q = apply_rope(queries.to(device=device, dtype=torch.float32), cos.unsqueeze(1), sin.unsqueeze(1))
    c = q.size(0)
    q = q.view(c, spec.num_kv_heads, spec.group_size, spec.head_dim)
    scores = torch.einsum("cgjd,gnd->cgjn", q, keys.to(torch.float32)) / math.sqrt(float(spec.head_dim))
    key_pos = torch.arange(keys.size(1), device=device)
    scores = scores.masked_fill(key_pos.view(1, 1, 1, -1) > pos.view(-1, 1, 1, 1), float("-inf"))
    return torch.softmax(scores, dim=-1).reshape(c, spec.num_q_heads, -1)


def aggregate_span_mass(attn_rows: torch.Tensor, token_to_span: torch.Tensor, num_spans: int) -> torch.Tensor:
    """Sum attention over each span's tokens: [..., seq_len] -> [..., num_spans]."""
    out = torch.zeros((*attn_rows.shape[:-1], num_spans + 1), dtype=torch.float32, device=attn_rows.device)
    out.index_add_(-1, token_to_span.to(attn_rows.device), attn_rows.to(torch.float32))
    return out[..., :num_spans]


def coverage_targets(span_mass: torch.Tensor, rho: float) -> torch.Tensor:
    """y_S = 1[S in C_rho]: the smallest set of spans whose cumulative mass reaches rho of the
    attention on those spans, computed independently for every leading index."""
    if rho <= 0.0 or rho > 1.0:
        raise ValueError(f"rho must be in (0,1], got {rho}")
    if span_mass.size(-1) == 0:
        return torch.zeros_like(span_mass, dtype=torch.float32)

    mass = span_mass.to(dtype=torch.float32).clamp_min(0.0)
    total = mass.sum(dim=-1, keepdim=True)
    norm = torch.where(total > 0, mass / total.clamp_min(1.0e-12), torch.zeros_like(mass))
    sorted_mass, order = torch.sort(norm, dim=-1, descending=True)
    before = torch.cumsum(sorted_mass, dim=-1) - sorted_mass
    keep_sorted = (before < float(rho)) & (sorted_mass > 0)
    y = torch.zeros_like(norm)
    y.scatter_(-1, order, keep_sorted.to(torch.float32))
    return y


@torch.no_grad()
def build_teacher_signals(
    backbone: FrozenBackbone,
    prefill: PrefillOutputs,
    seg: SegmentationResult,
    cfg: SeKVConfig,
    rope_fn: RopeFn,
    candidate_chunk: int = 16,
) -> TeacherSignals:
    """Select query positions whose teacher attention falls mostly outside the local window and
    collect the teacher's span attention mass and output logits there.

    `prefill` must have been run with keep_hidden=True and query_positions set to the candidates.
    """
    candidates = list(prefill.query_positions)
    if not candidates or prefill.queries is None or prefill.final_hidden is None:
        return TeacherSignals([], {}, {}, {})

    spec = backbone.spec
    window = int(cfg.memory.local_window)
    num_spans = len(seg.spans)
    token_to_span = token_span_index(seg).to(prefill.device)
    key_pos = torch.arange(seg.seq_len, device=prefill.device)

    far = torch.zeros(len(candidates), dtype=torch.float32)
    mass = torch.zeros((len(candidates), spec.num_layers, spec.num_q_heads, num_spans), dtype=torch.float32)

    for layer_idx in range(spec.num_layers):
        keys = prefill.keys[layer_idx]
        for c0 in range(0, len(candidates), candidate_chunk):
            chunk = candidates[c0 : c0 + candidate_chunk]
            rows = teacher_attention_rows(prefill.queries[layer_idx][c0 : c0 + len(chunk)], keys, chunk, spec, rope_fn)
            limit = torch.tensor([p - window + 1 for p in chunk], device=rows.device).view(-1, 1, 1)
            far_mass = (rows * (key_pos.view(1, 1, -1) < limit)).sum(dim=-1).mean(dim=-1)
            far[c0 : c0 + len(chunk)] += far_mass.cpu() / spec.num_layers
            mass[c0 : c0 + len(chunk), layer_idx] = aggregate_span_mass(rows, token_to_span, num_spans).cpu()

    k = min(int(cfg.train.queries_per_seq), len(candidates))
    top = torch.topk(far, k=k).indices.sort().values.tolist()
    selected = [candidates[i] for i in top]

    logits = backbone.lm_logits(prefill.final_hidden[torch.tensor(selected, device=prefill.final_hidden.device)])
    return TeacherSignals(
        query_positions=selected,
        span_mass={candidates[i]: mass[i] for i in top},
        far_fraction={candidates[i]: float(far[i]) for i in top},
        teacher_logits={pos: logits[j].detach().cpu() for j, pos in enumerate(selected)},
    )
