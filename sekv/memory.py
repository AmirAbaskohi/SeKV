from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sekv.backbone import ModelSpec, PrefillOutputs
from sekv.config import MemoryConfig, SeKVConfig
from sekv.segmentation import SegmentationResult


@dataclass
class SpanFactors:
    # CPU-resident detached factors for one span, stacked over KV heads.
    U_k: torch.Tensor
    s_k: torch.Tensor
    Vmat_k: torch.Tensor
    U_v: torch.Tensor
    s_v: torch.Tensor
    Vmat_v: torch.Tensor
    valid_rank: int


@dataclass
class SpanEntry:
    token_indices: list[int]
    length: int
    full_resolution: bool
    k_bar: torch.Tensor
    v_bar: torch.Tensor
    factors: Optional[SpanFactors]
    raw_k: Optional[torch.Tensor]
    raw_v: Optional[torch.Tensor]


@dataclass
class LayerMemory:
    anchor_k: torch.Tensor
    anchor_v: torch.Tensor
    anchor_indices: list[int]
    spans: list[SpanEntry]


@dataclass
class SpanMemory:
    spec: ModelSpec
    layers: list[LayerMemory]
    seg: SegmentationResult


def _as_device(device: torch.device | str) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def span_weights(surprisal: torch.Tensor, token_indices: list[int]) -> torch.Tensor:
    """Normalized surprisal weights over a span's tokens; uniform if sum==0."""
    if surprisal.ndim != 1:
        raise ValueError(f"surprisal must have shape [seq_len], got {tuple(surprisal.shape)}")
    if not token_indices:
        raise ValueError("token_indices must be non-empty")

    idx = torch.tensor(token_indices, dtype=torch.long, device=surprisal.device)
    span_surprisal = surprisal.index_select(dim=0, index=idx).to(dtype=torch.float32)
    total = span_surprisal.sum()

    if float(total.item()) <= 0.0:
        return torch.full(
            (len(token_indices),),
            1.0 / float(len(token_indices)),
            dtype=torch.float32,
            device=surprisal.device,
        )
    return span_surprisal / total


def _batched_lowrank_svd(
    mats: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mats.ndim != 3:
        raise ValueError(f"mats must have shape [B, M, N], got {tuple(mats.shape)}")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")

    m = mats.size(-2)
    n = mats.size(-1)
    max_rank = min(m, n)
    if rank > max_rank:
        raise ValueError(f"rank={rank} exceeds min(M,N)={max_rank}")

    def _one(mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, s, v = torch.svd_lowrank(mat, q=rank, niter=2)
        return u, s, v

    if hasattr(torch, "vmap"):
        u, s, v = torch.vmap(_one)(mats)
    elif hasattr(torch, "func") and hasattr(torch.func, "vmap"):
        u, s, v = torch.func.vmap(_one)(mats)
    else:
        raise RuntimeError("torch.vmap is required to compute SVD over KV heads without loops")

    return u, s, v


def build_layer_memory(
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
    seg: SegmentationResult,
    cfg: MemoryConfig,
    device,
    svd_device,
) -> LayerMemory:
    """Build one layer's dual-resolution memory from post-RoPE K and value states."""
    if cfg.r_max <= 0:
        raise ValueError(f"cfg.r_max must be > 0, got {cfg.r_max}")

    if layer_k.ndim != 3 or layer_v.ndim != 3:
        raise ValueError(
            f"layer_k/layer_v must have shape [H_kv, seq_len, d_h], got "
            f"K{tuple(layer_k.shape)} V{tuple(layer_v.shape)}"
        )
    if layer_k.shape != layer_v.shape:
        raise ValueError(
            f"layer_k and layer_v must match; got {tuple(layer_k.shape)} and {tuple(layer_v.shape)}"
        )

    h_kv, seq_len, d_h = layer_k.shape
    if seq_len != seg.seq_len:
        raise ValueError(
            f"Layer seq_len {seq_len} does not match segmentation seq_len {seg.seq_len}"
        )

    model_device = _as_device(device)
    svd_dev = _as_device(svd_device)

    layer_k_dev = layer_k.to(device=model_device)
    layer_v_dev = layer_v.to(device=model_device)

    if seg.anchor_indices:
        anchor_idx = torch.tensor(seg.anchor_indices, dtype=torch.long, device=model_device)
        # Anchor K/V are kept full-resolution on model device for decode-time access.
        anchor_k = layer_k_dev.index_select(dim=1, index=anchor_idx).contiguous()
        anchor_v = layer_v_dev.index_select(dim=1, index=anchor_idx).contiguous()
    else:
        anchor_k = layer_k_dev.new_empty((h_kv, 0, d_h))
        anchor_v = layer_v_dev.new_empty((h_kv, 0, d_h))

    spans: list[SpanEntry] = []
    for span in seg.spans:
        token_indices = list(span.token_indices)
        length = len(token_indices)
        if length <= 0:
            raise ValueError("Encountered empty span in segmentation result")

        idx = torch.tensor(token_indices, dtype=torch.long, device=model_device)
        span_k = layer_k_dev.index_select(dim=1, index=idx).contiguous()
        span_v = layer_v_dev.index_select(dim=1, index=idx).contiguous()

        weights = span_weights(seg.surprisal, token_indices).to(device=model_device, dtype=torch.float32)
        weights = weights.view(1, length, 1)

        span_k_f32 = span_k.to(dtype=torch.float32)
        span_v_f32 = span_v.to(dtype=torch.float32)

        # Weighted means are stored on model device for later routing/reconstruction usage.
        k_bar = (span_k_f32 * weights).sum(dim=1).contiguous()
        v_bar = (span_v_f32 * weights).sum(dim=1).contiguous()

        if span.full_resolution:
            # Full-resolution spans stay on model device and skip SVD.
            spans.append(
                SpanEntry(
                    token_indices=token_indices,
                    length=length,
                    full_resolution=True,
                    k_bar=k_bar,
                    v_bar=v_bar,
                    factors=None,
                    raw_k=span_k,
                    raw_v=span_v,
                )
            )
            continue

        valid_rank = min(length, d_h, int(cfg.r_max))
        if valid_rank <= 0:
            raise ValueError(f"Computed invalid rank {valid_rank} for span length {length}")

        U_k, s_k, Vmat_k = _batched_lowrank_svd(span_k_f32.to(device=svd_dev), rank=valid_rank)
        U_v, s_v, Vmat_v = _batched_lowrank_svd(span_v_f32.to(device=svd_dev), rank=valid_rank)

        # SVD factors are constants for training: detach and place on CPU.
        factors = SpanFactors(
            U_k=U_k.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            s_k=s_k.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            Vmat_k=Vmat_k.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            U_v=U_v.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            s_v=s_v.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            Vmat_v=Vmat_v.detach().to(device="cpu", dtype=torch.float32).contiguous(),
            valid_rank=valid_rank,
        )

        spans.append(
            SpanEntry(
                token_indices=token_indices,
                length=length,
                full_resolution=False,
                k_bar=k_bar,
                v_bar=v_bar,
                factors=factors,
                raw_k=None,
                raw_v=None,
            )
        )

    return LayerMemory(
        anchor_k=anchor_k,
        anchor_v=anchor_v,
        anchor_indices=list(seg.anchor_indices),
        spans=spans,
    )


def build_span_memory(
    prefill: PrefillOutputs,
    seg: SegmentationResult,
    cfg: SeKVConfig,
    spec: ModelSpec,
) -> SpanMemory:
    """Build full multi-layer span memory; GPU tensors on model device, SVD factors on CPU."""
    if len(prefill.keys) != spec.num_layers or len(prefill.values) != spec.num_layers:
        raise ValueError(
            "prefill layer count mismatch: "
            f"keys={len(prefill.keys)}, values={len(prefill.values)}, expected={spec.num_layers}"
        )

    model_device = prefill.next_token_logits.device
    svd_device_name = getattr(cfg.memory, "svd_device", "cpu")
    svd_device = _as_device(svd_device_name)

    layers: list[LayerMemory] = []
    for layer_idx in range(spec.num_layers):
        layer_k = prefill.keys[layer_idx]
        layer_v = prefill.values[layer_idx]

        if layer_k.ndim != 3 or layer_v.ndim != 3:
            raise ValueError(
                f"Layer {layer_idx} K/V must be rank-3 [H_kv, seq_len, d_h], got "
                f"K{tuple(layer_k.shape)} V{tuple(layer_v.shape)}"
            )
        if layer_k.shape != layer_v.shape:
            raise ValueError(
                f"Layer {layer_idx} K/V shape mismatch: {tuple(layer_k.shape)} vs {tuple(layer_v.shape)}"
            )
        if layer_k.size(0) != spec.num_kv_heads:
            raise ValueError(
                f"Layer {layer_idx} kv head mismatch: got {layer_k.size(0)}, expected {spec.num_kv_heads}"
            )
        if layer_k.size(1) != seg.seq_len:
            raise ValueError(
                f"Layer {layer_idx} seq_len mismatch: got {layer_k.size(1)}, expected {seg.seq_len}"
            )
        if layer_k.size(2) != spec.head_dim:
            raise ValueError(
                f"Layer {layer_idx} head_dim mismatch: got {layer_k.size(2)}, expected {spec.head_dim}"
            )

        layers.append(
            build_layer_memory(
                layer_k=layer_k,
                layer_v=layer_v,
                seg=seg,
                cfg=cfg.memory,
                device=model_device,
                svd_device=svd_device,
            )
        )

    return SpanMemory(spec=spec, layers=layers, seg=seg)


def reconstruct_span(
    factors_U: torch.Tensor,
    factors_s: torch.Tensor,
    factors_Vmat: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """Eq.11 gated reconstruction stacked over KV heads; differentiable in gates only."""
    if factors_U.ndim != 3 or factors_s.ndim != 2 or factors_Vmat.ndim != 3 or gates.ndim != 2:
        raise ValueError(
            "Expected U:[H_kv,|S|,R], s:[H_kv,R], Vmat:[H_kv,d_h,R], gates:[H_kv,R]"
        )

    h_kv, span_len, rank = factors_U.shape
    if factors_s.shape != (h_kv, rank):
        raise ValueError(f"factors_s shape mismatch: got {tuple(factors_s.shape)}, expected {(h_kv, rank)}")
    if gates.shape != (h_kv, rank):
        raise ValueError(f"gates shape mismatch: got {tuple(gates.shape)}, expected {(h_kv, rank)}")
    if factors_Vmat.size(0) != h_kv or factors_Vmat.size(2) != rank:
        raise ValueError(
            f"factors_Vmat shape mismatch: got {tuple(factors_Vmat.shape)}, expected [H_kv, d_h, R]"
        )

    U_const = factors_U.detach().to(dtype=torch.float32)
    s_const = factors_s.detach().to(device=U_const.device, dtype=torch.float32)
    V_const = factors_Vmat.detach().to(device=U_const.device, dtype=torch.float32)
    gates_f32 = gates.to(device=U_const.device, dtype=torch.float32)

    scaled = gates_f32 * s_const
    left = U_const * scaled.unsqueeze(1)
    return torch.einsum("hsr,hdr->hsd", left, V_const)


def effective_rank(gates: torch.Tensor) -> torch.Tensor:
    """Per-kv-head effective rank r_S = sum_i gates_i."""
    if gates.ndim != 2:
        raise ValueError(f"gates must have shape [H_kv, R], got {tuple(gates.shape)}")
    return gates.to(dtype=torch.float32).sum(dim=-1)
