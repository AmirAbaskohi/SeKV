from __future__ import annotations

from dataclasses import dataclass

import torch

from sekv.backbone import PrefillOutputs
from sekv.config import SegmentationConfig


@dataclass(frozen=True)
class Span:
    """Contiguous tokens [start, stop) between two consecutive anchors."""

    start: int
    stop: int
    full_resolution: bool

    @property
    def length(self) -> int:
        return self.stop - self.start

    @property
    def last(self) -> int:
        return self.stop - 1

    @property
    def token_indices(self) -> list[int]:
        return list(range(self.start, self.stop))


@dataclass(frozen=True)
class SegmentationResult:
    seq_len: int
    surprisal: torch.Tensor
    anchor_indices: list[int]
    spans: list[Span]
    boundary_threshold: float


@dataclass(frozen=True)
class MemoryLayout:
    """Causal view of a segmented sequence for a query at `query_position`.

    Anchors and spans that lie entirely before the local window form a prefix of the
    segmentation (`num_anchors`, `num_spans`); every token in [window_start, query_position]
    is kept as exact KV in the local window.
    """

    num_anchors: int
    num_spans: int
    window_start: int
    query_position: int


def _validate_partition(seq_len: int, anchors: list[int], spans: list[Span], l_min: int) -> None:
    if anchors != sorted(set(anchors)):
        raise ValueError("anchor_indices must be strictly ascending")

    elements: list[tuple[int, int]] = [(a, a + 1) for a in anchors]
    for idx, span in enumerate(spans):
        if span.length <= 0:
            raise ValueError(f"Span {idx} is empty; empty spans are not allowed")
        if span.full_resolution != (span.length < l_min):
            raise ValueError(
                f"Span {idx} full_resolution mismatch: expected {span.length < l_min}, "
                f"got {span.full_resolution}"
            )
        elements.append((span.start, span.stop))

    elements.sort()
    cursor = 0
    for start, stop in elements:
        if start != cursor:
            raise ValueError(f"Invalid partition of sequence positions at token {cursor}")
        cursor = stop
    if cursor != seq_len:
        raise ValueError(f"Invalid partition: covered {cursor} of {seq_len} positions")


def _merge_close_boundaries(candidates: list[int], surprisal: torch.Tensor, l_gap: int) -> list[int]:
    """Keep the highest-surprisal boundary among any boundaries closer than l_gap tokens."""
    if l_gap <= 1 or not candidates:
        return sorted(candidates)

    values = surprisal.tolist()
    order = sorted(candidates, key=lambda pos: values[pos], reverse=True)
    blocked = bytearray(len(values))
    kept: list[int] = []
    for pos in order:
        if blocked[pos]:
            continue
        kept.append(pos)
        lo = max(0, pos - l_gap + 1)
        hi = min(len(values), pos + l_gap)
        blocked[lo:hi] = b"\x01" * (hi - lo)
    return sorted(kept)


def compute_surprisal(prefill: PrefillOutputs, input_ids: torch.Tensor) -> torch.Tensor:
    """Return [seq_len] surprisal in nats, H_t = -log p(x_t | x_<t), with H_0=0."""
    ids = input_ids.reshape(-1)
    if int(ids.numel()) != prefill.seq_len:
        raise ValueError(
            f"Prefill length ({prefill.seq_len}) does not match input_ids length ({ids.numel()})"
        )
    return prefill.surprisal.to(device="cpu", dtype=torch.float32)


def segment_sequence(surprisal: torch.Tensor, cfg: SegmentationConfig) -> SegmentationResult:
    """Mark H_t > mu + alpha*sigma as boundaries, merge boundaries closer than l_gap, and emit
    anchors (boundary tokens) plus the spans between them. Position 0 starts the first segment
    and is always an anchor. Spans shorter than l_min are flagged full_resolution."""
    if surprisal.ndim != 1:
        raise ValueError(f"surprisal must have shape [seq_len], got {tuple(surprisal.shape)}")
    if cfg.l_min < 0 or cfg.l_gap < 0:
        raise ValueError(f"l_min and l_gap must be >= 0, got {cfg.l_min} and {cfg.l_gap}")

    surprisal_cpu = surprisal.detach().to(device="cpu", dtype=torch.float32).contiguous()
    seq_len = int(surprisal_cpu.numel())

    if seq_len == 0:
        return SegmentationResult(0, surprisal_cpu, [], [], 0.0)

    mu = float(surprisal_cpu.mean().item())
    sigma = float(surprisal_cpu.std(unbiased=False).item())
    boundary_threshold = mu + float(cfg.alpha) * sigma

    candidates = (surprisal_cpu > boundary_threshold).nonzero(as_tuple=False).squeeze(-1).tolist()
    candidates = [int(pos) for pos in candidates if pos > 0]
    anchors = [0] + _merge_close_boundaries(candidates, surprisal_cpu, int(cfg.l_gap))

    spans: list[Span] = []
    for idx, anchor in enumerate(anchors):
        start = anchor + 1
        stop = anchors[idx + 1] if idx + 1 < len(anchors) else seq_len
        if start < stop:
            spans.append(Span(start=start, stop=stop, full_resolution=(stop - start) < cfg.l_min))

    _validate_partition(seq_len, anchors, spans, cfg.l_min)

    return SegmentationResult(
        seq_len=seq_len,
        surprisal=surprisal_cpu,
        anchor_indices=anchors,
        spans=spans,
        boundary_threshold=float(boundary_threshold),
    )


def segment_prefill(
    prefill: PrefillOutputs,
    input_ids: torch.Tensor,
    cfg: SegmentationConfig,
) -> SegmentationResult:
    """Convenience: compute_surprisal then segment_sequence."""
    surprisal = compute_surprisal(prefill=prefill, input_ids=input_ids)
    return segment_sequence(surprisal=surprisal, cfg=cfg)


def memory_layout(seg: SegmentationResult, query_position: int, local_window: int) -> MemoryLayout:
    """Split the context of a query into compressed memory and the exact local window.

    Elements (anchors and spans) that end before `query_position - local_window + 1` stay in
    span memory; the remaining tokens up to and including the query form the local window.
    """
    if query_position < 0 or query_position >= seg.seq_len:
        raise ValueError(f"query_position {query_position} out of range for seq_len={seg.seq_len}")
    if local_window < 1:
        raise ValueError(f"local_window must be >= 1, got {local_window}")

    limit = max(0, query_position - local_window + 1)
    num_anchors = sum(1 for a in seg.anchor_indices if a < limit)
    num_spans = sum(1 for span in seg.spans if span.stop <= limit)

    covered = -1
    if num_anchors:
        covered = max(covered, seg.anchor_indices[num_anchors - 1])
    if num_spans:
        covered = max(covered, seg.spans[num_spans - 1].last)

    return MemoryLayout(
        num_anchors=num_anchors,
        num_spans=num_spans,
        window_start=covered + 1,
        query_position=int(query_position),
    )
