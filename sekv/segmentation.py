from __future__ import annotations

from dataclasses import dataclass

import torch

from sekv.backbone import PrefillOutputs
from sekv.config import SegmentationConfig


@dataclass(frozen=True)
class Span:
    token_indices: list[int]
    full_resolution: bool


@dataclass(frozen=True)
class SegmentationResult:
    seq_len: int
    surprisal: torch.Tensor
    anchor_indices: list[int]
    spans: list[Span]
    boundary_threshold: float


def _normalize_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim == 2:
        if input_ids.size(0) != 1:
            raise ValueError(
                f"input_ids must be shape [seq_len] or [1, seq_len], got {tuple(input_ids.shape)}"
            )
        input_ids = input_ids[0]
    elif input_ids.ndim != 1:
        raise ValueError(
            f"input_ids must be shape [seq_len] or [1, seq_len], got {tuple(input_ids.shape)}"
        )

    if input_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"input_ids must be an integer tensor, got dtype {input_ids.dtype}")

    return input_ids.contiguous()


def _validate_spans(spans: list[Span], l_min: int) -> None:
    prev_last = -1
    for idx, span in enumerate(spans):
        token_indices = span.token_indices
        if not token_indices:
            raise ValueError(f"Span {idx} is empty; empty spans are not allowed")

        for pos in token_indices:
            if not isinstance(pos, int):
                raise TypeError(f"Span {idx} contains non-int token index: {type(pos).__name__}")

        start = token_indices[0]
        end = token_indices[-1]
        expected = list(range(start, end + 1))
        if token_indices != expected:
            raise ValueError(f"Span {idx} token_indices must be contiguous and ascending")

        if start <= prev_last:
            raise ValueError(f"Spans overlap or are not strictly ascending at span {idx}")

        expected_full_resolution = len(token_indices) < l_min
        if span.full_resolution != expected_full_resolution:
            raise ValueError(
                f"Span {idx} full_resolution mismatch: expected {expected_full_resolution}, "
                f"got {span.full_resolution}"
            )

        prev_last = end


def _assert_exact_partition(seq_len: int, anchors: list[int], spans: list[Span]) -> None:
    anchor_set = set(anchors)

    if len(anchor_set) != len(anchors):
        raise ValueError("anchor_indices must not contain duplicates")
    if anchors != sorted(anchors):
        raise ValueError("anchor_indices must be ascending")
    for anchor in anchors:
        if anchor < 0 or anchor >= seq_len:
            raise ValueError(f"Anchor index out of bounds: {anchor}")

    span_positions: set[int] = set()
    for idx, span in enumerate(spans):
        for pos in span.token_indices:
            if pos < 0 or pos >= seq_len:
                raise ValueError(f"Span {idx} contains out-of-bounds position: {pos}")
            if pos in span_positions:
                raise ValueError(f"Span positions overlap at token index {pos}")
            span_positions.add(pos)

    if span_positions & anchor_set:
        overlap = min(span_positions & anchor_set)
        raise ValueError(f"Spans and anchors overlap at token index {overlap}")

    # Zero-boundary convention: when no anchors are detected, SeKV may keep position 0
    # outside span decomposition and segment only positions [1..seq_len-1].
    if seq_len > 0 and not anchors:
        if (
            len(spans) == 1
            and spans[0].token_indices
            and spans[0].token_indices[0] == 1
            and spans[0].token_indices[-1] == seq_len - 1
        ):
            anchor_set = {0}

    covered = span_positions | anchor_set
    expected = set(range(seq_len))
    if covered != expected:
        missing = sorted(expected - covered)
        extra = sorted(covered - expected)
        raise ValueError(
            f"Invalid partition of sequence positions. Missing={missing}, Extra={extra}"
        )


def compute_surprisal(prefill: PrefillOutputs, input_ids: torch.Tensor) -> torch.Tensor:
    """Return [seq_len] surprisal in nats, H_t = -log p(realized token t | t_<t), with H_0=0."""
    ids = _normalize_input_ids(input_ids)
    seq_len = int(ids.numel())

    logits = prefill.next_token_logits
    if logits.ndim != 2:
        raise ValueError(
            f"prefill.next_token_logits must have shape [seq_len, vocab], got {tuple(logits.shape)}"
        )
    if logits.size(0) != seq_len:
        raise ValueError(
            f"Logit length ({logits.size(0)}) does not match input_ids length ({seq_len})"
        )

    surprisal = torch.zeros(seq_len, dtype=torch.float32, device="cpu")
    if seq_len <= 1:
        return surprisal

    ids_for_gather = ids.to(device=logits.device, dtype=torch.long)
    log_probs = torch.log_softmax(logits.to(dtype=torch.float32), dim=-1)
    realized_log_probs = log_probs[:-1].gather(dim=1, index=ids_for_gather[1:].unsqueeze(1)).squeeze(1)

    surprisal[1:] = (-realized_log_probs).to(dtype=torch.float32, device="cpu")
    return surprisal


def segment_sequence(surprisal: torch.Tensor, cfg: SegmentationConfig) -> SegmentationResult:
    """Threshold at mu + alpha*sigma, emit anchors and the spans between them, flag spans shorter than l_min as full_resolution."""
    if surprisal.ndim != 1:
        raise ValueError(f"surprisal must have shape [seq_len], got {tuple(surprisal.shape)}")

    surprisal_cpu = surprisal.detach().to(device="cpu", dtype=torch.float32).contiguous()
    seq_len = int(surprisal_cpu.numel())

    if cfg.l_min < 0:
        raise ValueError(f"cfg.l_min must be >= 0, got {cfg.l_min}")

    if seq_len == 0:
        return SegmentationResult(
            seq_len=0,
            surprisal=surprisal_cpu,
            anchor_indices=[],
            spans=[],
            boundary_threshold=0.0,
        )

    mu = float(surprisal_cpu.mean().item())
    sigma = float(surprisal_cpu.std(unbiased=False).item())
    boundary_threshold = mu + float(cfg.alpha) * sigma

    anchor_mask = surprisal_cpu > boundary_threshold
    anchor_indices = anchor_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    spans: list[Span] = []

    if not anchor_indices:
        if seq_len < cfg.l_min:
            all_indices = list(range(seq_len))
            spans.append(Span(token_indices=all_indices, full_resolution=True))
        else:
            if seq_len > 1:
                span_indices = list(range(1, seq_len))
                spans.append(
                    Span(token_indices=span_indices, full_resolution=(len(span_indices) < cfg.l_min))
                )
            elif seq_len == 1:
                spans.append(Span(token_indices=[0], full_resolution=(1 < cfg.l_min)))
    else:
        prev_anchor = -1
        for anchor in anchor_indices:
            start = prev_anchor + 1
            end = anchor - 1
            if start <= end:
                token_indices = list(range(start, end + 1))
                spans.append(
                    Span(token_indices=token_indices, full_resolution=(len(token_indices) < cfg.l_min))
                )
            prev_anchor = anchor

        trail_start = prev_anchor + 1
        if trail_start <= seq_len - 1:
            token_indices = list(range(trail_start, seq_len))
            spans.append(
                Span(token_indices=token_indices, full_resolution=(len(token_indices) < cfg.l_min))
            )

    _validate_spans(spans, cfg.l_min)
    _assert_exact_partition(seq_len=seq_len, anchors=anchor_indices, spans=spans)

    return SegmentationResult(
        seq_len=seq_len,
        surprisal=surprisal_cpu,
        anchor_indices=[int(idx) for idx in anchor_indices],
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
