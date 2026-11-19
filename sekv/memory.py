from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import torch

from sekv.backbone import ModelSpec, PrefillOutputs
from sekv.config import MemoryConfig, SeKVConfig
from sekv.modules import KIND_KEY, KIND_VALUE, SeKVModules
from sekv.rope import RopeFn, apply_rope, invert_rope
from sekv.segmentation import MemoryLayout, SegmentationResult, Span


@dataclass
class SpanFactors:
    """Low-rank factors of one span's pre-RoPE keys and values, stacked over KV heads.

    X_S ~= U diag(s) Vmat^T with U:[H_kv,|S|,r], s:[H_kv,r], Vmat:[H_kv,d_h,r]. When `gated` is
    True the learned gates are already folded into s and components with negligible gates were
    dropped, so rank_k / rank_v hold the stored (learned) rank per KV head.
    """

    U_k: torch.Tensor
    s_k: torch.Tensor
    Vmat_k: torch.Tensor
    U_v: torch.Tensor
    s_v: torch.Tensor
    Vmat_v: torch.Tensor
    rank_k: list[int]
    rank_v: list[int]
    energy_k: torch.Tensor
    energy_v: torch.Tensor
    gated: bool

    @property
    def valid_rank(self) -> int:
        return int(max(self.s_k.size(-1), self.s_v.size(-1)))

    def group_nbytes(self, group: int) -> int:
        span_len = int(self.U_k.size(1))
        d_h = int(self.Vmat_k.size(1))
        per = self.U_k.element_size()
        comps = int(self.rank_k[group]) + int(self.rank_v[group])
        return comps * (span_len + d_h + 1) * per

    def nbytes(self) -> int:
        return sum(self.group_nbytes(g) for g in range(len(self.rank_k)))


@dataclass
class SpanEntry:
    start: int
    length: int
    full_resolution: bool
    # Surprisal-weighted mean of post-RoPE keys and of values, [H_kv, d_h] (the coarse entry).
    k_bar: torch.Tensor
    v_bar: torch.Tensor
    factors: Optional[SpanFactors]
    # Exact post-RoPE keys / values for spans shorter than l_min, [H_kv, |S|, d_h].
    raw_k: Optional[torch.Tensor]
    raw_v: Optional[torch.Tensor]

    @property
    def stop(self) -> int:
        return self.start + self.length

    @property
    def token_indices(self) -> list[int]:
        return list(range(self.start, self.stop))


@dataclass
class LayerDerived:
    """Stacked tensors derived from a layer's static memory, reused across decoding steps."""

    comp_ids: list[int]
    k_bar: torch.Tensor
    v_bar: torch.Tensor
    lengths: torch.Tensor
    anchor_k: torch.Tensor
    anchor_v: torch.Tensor
    fr_ids: list[int]
    fr_k: torch.Tensor
    fr_v: torch.Tensor
    fr_cum: list[int]

    @cached_property
    def exact_k(self) -> torch.Tensor:
        return torch.cat([self.anchor_k, self.fr_k], dim=1)

    @cached_property
    def exact_v(self) -> torch.Tensor:
        return torch.cat([self.anchor_v, self.fr_v], dim=1)

    @property
    def num_exact(self) -> int:
        return int(self.anchor_k.size(1) + self.fr_k.size(1))


@dataclass
class LayerMemory:
    anchor_k: torch.Tensor
    anchor_v: torch.Tensor
    anchor_indices: list[int]
    spans: list[SpanEntry]
    window_k: torch.Tensor
    window_v: torch.Tensor
    window_start: int
    cache: dict = field(default_factory=dict)

    @property
    def window_len(self) -> int:
        return int(self.window_k.size(1))

    def append_window(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append one token's post-RoPE key and value ([H_kv, d_h]) to the local window."""
        self.window_k = torch.cat([self.window_k, k.to(self.window_k).unsqueeze(1)], dim=1)
        self.window_v = torch.cat([self.window_v, v.to(self.window_v).unsqueeze(1)], dim=1)

    def invalidate(self) -> None:
        self.cache.clear()


@dataclass
class SpanMemory:
    spec: ModelSpec
    layers: list[LayerMemory]
    seg: SegmentationResult

    @property
    def window_start(self) -> int:
        return self.layers[0].window_start if self.layers else 0


def span_weights(surprisal: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Surprisal weights w_t = H_t / sum_k H_k over [start, stop); uniform if the sum is zero."""
    if surprisal.ndim != 1:
        raise ValueError(f"surprisal must have shape [seq_len], got {tuple(surprisal.shape)}")
    if stop <= start:
        raise ValueError("span must be non-empty")
    h = surprisal[start:stop].to(dtype=torch.float32).clamp_min(0.0)
    total = float(h.sum().item())
    if total <= 0.0:
        return torch.full((stop - start,), 1.0 / float(stop - start), dtype=torch.float32)
    return h / total


def _compact_gated(
    U: torch.Tensor,
    S: torch.Tensor,
    Vmat: torch.Tensor,
    gates: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold gates into S, drop components with gate <= eps, and move kept ones to the front."""
    keep = (gates > eps) & (S > 0)
    # Keep at least the leading component so a selected span never reconstructs to zero.
    keep[..., 0] |= (~keep.any(dim=-1)) & (S[..., 0] > 0)
    order = torch.argsort((~keep).to(torch.int32), dim=-1, stable=True)
    keep_sorted = keep.gather(-1, order)
    S_out = (S * gates).gather(-1, order) * keep_sorted.to(S.dtype)
    U_out = U.gather(-1, order.unsqueeze(-2).expand_as(U))
    V_out = Vmat.gather(-1, order.unsqueeze(-2).expand_as(Vmat))
    return U_out, S_out, V_out, keep.sum(dim=-1)


def build_span_entries(
    layer_k: torch.Tensor,
    layer_v: torch.Tensor,
    spans: list[Span],
    surprisal: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cfg: MemoryConfig,
    offset: int = 0,
    modules: Optional[SeKVModules] = None,
    layer: int = 0,
    offload: bool = False,
    bucket_size: int = 32,
) -> list[SpanEntry]:
    """Build dual-resolution entries for spans whose tokens live in layer_k/layer_v.

    layer_k/layer_v: [H_kv, n, d_h] post-RoPE keys / values for absolute positions offset..offset+n-1.
    cos/sin: [n, d_h] RoPE tables for the same positions. With `modules`, learned rank gates are
    applied and negligible components dropped (inference storage); otherwise full factors are kept
    so gates can be trained.
    """
    del layer
    if cfg.r_max <= 0:
        raise ValueError(f"cfg.r_max must be > 0, got {cfg.r_max}")
    if layer_k.shape != layer_v.shape or layer_k.ndim != 3:
        raise ValueError(f"layer_k/layer_v must match [H_kv, n, d_h], got {tuple(layer_k.shape)} / {tuple(layer_v.shape)}")

    h_kv, _, d_h = layer_k.shape
    device = layer_k.device
    entries: list[Optional[SpanEntry]] = [None] * len(spans)

    def _local(span: Span) -> slice:
        return slice(span.start - offset, span.stop - offset)

    def _means(span: Span, k_post: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = span_weights(surprisal, span.start, span.stop).to(device=device).view(1, -1, 1)
        return (k_post * w).sum(dim=1), (v * w).sum(dim=1)

    compressed: list[int] = []
    for idx, span in enumerate(spans):
        if span.start < offset or span.stop - offset > layer_k.size(1):
            raise ValueError(f"Span [{span.start},{span.stop}) is outside the provided KV range")
        if not span.full_resolution:
            compressed.append(idx)
            continue
        sl = _local(span)
        raw_k = layer_k[:, sl].contiguous()
        raw_v = layer_v[:, sl].contiguous()
        k_bar, v_bar = _means(span, raw_k.to(torch.float32), raw_v.to(torch.float32))
        entries[idx] = SpanEntry(span.start, span.length, True, k_bar, v_bar, None, raw_k, raw_v)

    compressed.sort(key=lambda i: spans[i].length)
    for b0 in range(0, len(compressed), bucket_size):
        chunk = compressed[b0 : b0 + bucket_size]
        nb = len(chunk)
        lmax = max(spans[i].length for i in chunk)

        k_pre = torch.zeros((nb, h_kv, lmax, d_h), dtype=torch.float32, device=device)
        v_pad = torch.zeros_like(k_pre)
        k_bars = torch.zeros((nb, h_kv, d_h), dtype=torch.float32, device=device)
        v_bars = torch.zeros_like(k_bars)
        valid = torch.zeros((nb,), dtype=torch.long, device=device)

        for j, idx in enumerate(chunk):
            span = spans[idx]
            sl = _local(span)
            k_post = layer_k[:, sl].to(torch.float32)
            v = layer_v[:, sl].to(torch.float32)
            # Factorize pre-RoPE keys: RoPE is re-applied at the original positions on reconstruction.
            k_pre[j, :, : span.length] = invert_rope(k_post, cos[sl], sin[sl])
            v_pad[j, :, : span.length] = v
            k_bars[j], v_bars[j] = _means(span, k_post, v)
            valid[j] = min(span.length, d_h, int(cfg.r_max))

        rank = min(int(cfg.r_max), lmax, d_h)
        factors_kv = []
        for mats in (k_pre, v_pad):
            U, S, Vh = torch.linalg.svd(mats.reshape(nb * h_kv, lmax, d_h), full_matrices=False)
            U = U[..., :rank].reshape(nb, h_kv, lmax, rank)
            S = S[..., :rank].reshape(nb, h_kv, rank)
            Vmat = Vh[..., :rank, :].transpose(-1, -2).reshape(nb, h_kv, d_h, rank)
            S = S * (torch.arange(rank, device=device).view(1, 1, -1) < valid.view(-1, 1, 1)).to(S.dtype)
            energy = mats.pow(2).sum(dim=(-1, -2))
            factors_kv.append((U, S, Vmat, energy))

        (U_k, S_k, V_k, e_k), (U_v, S_v, V_v, e_v) = factors_kv
        if modules is not None:
            with torch.no_grad():
                g_k = modules.rank_gates(S_k, k_bars, valid.view(-1, 1), KIND_KEY)[..., :rank]
                g_v = modules.rank_gates(S_v, k_bars, valid.view(-1, 1), KIND_VALUE)[..., :rank]
            U_k, S_k, V_k, r_k = _compact_gated(U_k, S_k, V_k, g_k, float(cfg.gate_eps))
            U_v, S_v, V_v, r_v = _compact_gated(U_v, S_v, V_v, g_v, float(cfg.gate_eps))
        else:
            r_k = valid.view(-1, 1).expand(nb, h_kv)
            r_v = r_k

        store_device = torch.device("cpu") if offload else device
        U_k, S_k, V_k, U_v, S_v, V_v = (t.to(store_device) for t in (U_k, S_k, V_k, U_v, S_v, V_v))
        e_k, e_v, r_k, r_v = e_k.to(store_device), e_v.to(store_device), r_k.tolist(), r_v.tolist()

        for j, idx in enumerate(chunk):
            span = spans[idx]
            rk = max(1, max(r_k[j]))
            rv = max(1, max(r_v[j]))
            L = span.length
            factors = SpanFactors(
                U_k=U_k[j, :, :L, :rk].contiguous().clone(),
                s_k=S_k[j, :, :rk].contiguous().clone(),
                Vmat_k=V_k[j, :, :, :rk].contiguous().clone(),
                U_v=U_v[j, :, :L, :rv].contiguous().clone(),
                s_v=S_v[j, :, :rv].contiguous().clone(),
                Vmat_v=V_v[j, :, :, :rv].contiguous().clone(),
                rank_k=[int(r) for r in r_k[j]],
                rank_v=[int(r) for r in r_v[j]],
                energy_k=e_k[j].clone(),
                energy_v=e_v[j].clone(),
                gated=modules is not None,
            )
            entries[idx] = SpanEntry(span.start, L, False, k_bars[j].clone(), v_bars[j].clone(), factors, None, None)

    return [e for e in entries if e is not None]


def build_span_memory(
    prefill: PrefillOutputs,
    seg: SegmentationResult,
    layout: MemoryLayout,
    cfg: SeKVConfig,
    spec: ModelSpec,
    rope_fn: RopeFn,
    modules: Optional[SeKVModules] = None,
    offload: Optional[bool] = None,
    window_stop: Optional[int] = None,
) -> SpanMemory:
    """Build multi-layer span memory for a layout.

    Anchors and spans before the local window form the memory; tokens [layout.window_start,
    window_stop) from the prefill are the exact local window (window_stop defaults to the query
    position + 1). With `modules` the learned rank is applied and factors are offloaded to CPU.
    """
    if len(prefill.keys) != spec.num_layers or len(prefill.values) != spec.num_layers:
        raise ValueError(
            f"prefill layer count mismatch: keys={len(prefill.keys)}, values={len(prefill.values)}, "
            f"expected={spec.num_layers}"
        )
    if prefill.seq_len != seg.seq_len:
        raise ValueError(f"prefill seq_len {prefill.seq_len} does not match segmentation {seg.seq_len}")

    stop = layout.query_position + 1 if window_stop is None else int(window_stop)
    if stop < layout.window_start:
        raise ValueError(f"window_stop {stop} precedes window_start {layout.window_start}")
    do_offload = (modules is not None) if offload is None else bool(offload)

    device = prefill.device
    spans = seg.spans[: layout.num_spans]
    anchors = seg.anchor_indices[: layout.num_anchors]
    span_end = spans[-1].stop if spans else 0
    cos, sin = rope_fn(torch.arange(span_end, device=device)) if span_end else (None, None)

    layers: list[LayerMemory] = []
    for layer_idx in range(spec.num_layers):
        layer_k = prefill.keys[layer_idx]
        layer_v = prefill.values[layer_idx]
        if layer_k.shape != (spec.num_kv_heads, seg.seq_len, spec.head_dim) or layer_v.shape != layer_k.shape:
            raise ValueError(
                f"Layer {layer_idx} K/V must be [{spec.num_kv_heads}, {seg.seq_len}, {spec.head_dim}], "
                f"got K{tuple(layer_k.shape)} V{tuple(layer_v.shape)}"
            )

        anchor_idx = torch.tensor(anchors, dtype=torch.long, device=device)
        entries = (
            build_span_entries(
                layer_k[:, :span_end],
                layer_v[:, :span_end],
                spans,
                seg.surprisal,
                cos,
                sin,
                cfg.memory,
                offset=0,
                modules=modules,
                layer=layer_idx,
                offload=do_offload,
            )
            if spans
            else []
        )
        layers.append(
            LayerMemory(
                anchor_k=layer_k.index_select(1, anchor_idx).contiguous(),
                anchor_v=layer_v.index_select(1, anchor_idx).contiguous(),
                anchor_indices=list(anchors),
                spans=entries,
                window_k=layer_k[:, layout.window_start : stop].contiguous(),
                window_v=layer_v[:, layout.window_start : stop].contiguous(),
                window_start=layout.window_start,
            )
        )

    return SpanMemory(spec=spec, layers=layers, seg=seg)


def derive_layer(layer_mem: LayerMemory) -> LayerDerived:
    """Stack coarse entries and exact blocks for a layer (cached until the memory changes)."""
    key = (len(layer_mem.spans), int(layer_mem.anchor_k.size(1)))
    cached = layer_mem.cache.get("derived")
    if cached is not None and cached[0] == key:
        return cached[1]

    h_kv, _, d_h = layer_mem.anchor_k.shape
    device = layer_mem.anchor_k.device
    comp_ids = [i for i, s in enumerate(layer_mem.spans) if not s.full_resolution]
    fr_ids = [i for i, s in enumerate(layer_mem.spans) if s.full_resolution]

    if comp_ids:
        k_bar = torch.stack([layer_mem.spans[i].k_bar for i in comp_ids], dim=0).to(device=device, dtype=torch.float32)
        v_bar = torch.stack([layer_mem.spans[i].v_bar for i in comp_ids], dim=0).to(device=device, dtype=torch.float32)
    else:
        k_bar = torch.zeros((0, h_kv, d_h), dtype=torch.float32, device=device)
        v_bar = torch.zeros_like(k_bar)
    lengths = torch.tensor([layer_mem.spans[i].length for i in comp_ids], dtype=torch.float32, device=device)

    if fr_ids:
        fr_k = torch.cat([layer_mem.spans[i].raw_k for i in fr_ids], dim=1).to(layer_mem.anchor_k)
        fr_v = torch.cat([layer_mem.spans[i].raw_v for i in fr_ids], dim=1).to(layer_mem.anchor_v)
    else:
        fr_k = layer_mem.anchor_k.new_zeros((h_kv, 0, d_h))
        fr_v = layer_mem.anchor_v.new_zeros((h_kv, 0, d_h))
    fr_cum: list[int] = []
    total = 0
    for i in fr_ids:
        total += layer_mem.spans[i].length
        fr_cum.append(total)

    derived = LayerDerived(
        comp_ids=comp_ids,
        k_bar=k_bar,
        v_bar=v_bar,
        lengths=lengths,
        anchor_k=layer_mem.anchor_k,
        anchor_v=layer_mem.anchor_v,
        fr_ids=fr_ids,
        fr_k=fr_k,
        fr_v=fr_v,
        fr_cum=fr_cum,
    )
    layer_mem.cache["derived"] = (key, derived)
    return derived


def _slice_derived(base: LayerDerived, num_anchors: int, num_spans: int) -> LayerDerived:
    m = sum(1 for i in base.comp_ids if i < num_spans)
    f = sum(1 for i in base.fr_ids if i < num_spans)
    fr_tokens = base.fr_cum[f - 1] if f else 0
    return LayerDerived(
        comp_ids=base.comp_ids[:m],
        k_bar=base.k_bar[:m],
        v_bar=base.v_bar[:m],
        lengths=base.lengths[:m],
        anchor_k=base.anchor_k[:, :num_anchors],
        anchor_v=base.anchor_v[:, :num_anchors],
        fr_ids=base.fr_ids[:f],
        fr_k=base.fr_k[:, :fr_tokens],
        fr_v=base.fr_v[:, :fr_tokens],
        fr_cum=base.fr_cum[:f],
    )


def memory_view(
    memory: SpanMemory,
    layout: MemoryLayout,
    prefill: PrefillOutputs,
    window_stop: Optional[int] = None,
) -> SpanMemory:
    """Causal view of a memory built for a later query: keeps the anchor/span prefix of `layout`
    and takes the exact window [layout.window_start, window_stop) from the prefill K/V."""
    stop = layout.query_position + 1 if window_stop is None else int(window_stop)
    layers: list[LayerMemory] = []
    for layer_idx, full in enumerate(memory.layers):
        if layout.num_spans > len(full.spans) or layout.num_anchors > full.anchor_k.size(1):
            raise ValueError("memory_view layout exceeds the memory it is derived from")
        view = LayerMemory(
            anchor_k=full.anchor_k[:, : layout.num_anchors],
            anchor_v=full.anchor_v[:, : layout.num_anchors],
            anchor_indices=full.anchor_indices[: layout.num_anchors],
            spans=full.spans[: layout.num_spans],
            window_k=prefill.keys[layer_idx][:, layout.window_start : stop],
            window_v=prefill.values[layer_idx][:, layout.window_start : stop],
            window_start=layout.window_start,
        )
        key = (layout.num_spans, layout.num_anchors)
        view.cache["derived"] = (key, _slice_derived(derive_layer(full), layout.num_anchors, layout.num_spans))
        layers.append(view)
    return SpanMemory(spec=memory.spec, layers=layers, seg=memory.seg)


def roll_window(
    memory: SpanMemory,
    num_tokens: int,
    surprisal: torch.Tensor,
    cfg: SeKVConfig,
    rope_fn: RopeFn,
    modules: Optional[SeKVModules],
    offload: bool,
) -> None:
    """Move the oldest `num_tokens` window tokens into a new span in every layer.

    Keeps the local window bounded during generation without evicting anything: tokens that
    leave the window stay in attention as a span (coarse entry + low-rank factors).
    """
    if num_tokens <= 0:
        return
    for layer_idx, layer_mem in enumerate(memory.layers):
        if num_tokens > layer_mem.window_len:
            raise ValueError("Cannot roll more tokens than the window holds")
        start = layer_mem.window_start
        span = Span(start=start, stop=start + num_tokens, full_resolution=num_tokens < cfg.segmentation.l_min)
        positions = torch.arange(start, start + num_tokens, device=layer_mem.window_k.device)
        cos, sin = rope_fn(positions)
        entries = build_span_entries(
            layer_mem.window_k[:, :num_tokens],
            layer_mem.window_v[:, :num_tokens],
            [span],
            surprisal,
            cos,
            sin,
            cfg.memory,
            offset=start,
            modules=modules,
            layer=layer_idx,
            offload=offload,
        )
        layer_mem.spans.extend(entries)
        layer_mem.window_k = layer_mem.window_k[:, num_tokens:].contiguous()
        layer_mem.window_v = layer_mem.window_v[:, num_tokens:].contiguous()
        layer_mem.window_start = start + num_tokens
        layer_mem.invalidate()


def reconstruct_span(
    factors_U: torch.Tensor,
    factors_s: torch.Tensor,
    factors_Vmat: torch.Tensor,
    gates: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Eq.2: X_hat = sum_i m_i sigma_i a_i b_i^T over leading dims; differentiable in gates only."""
    U = factors_U.detach().to(dtype=torch.float32)
    s = factors_s.detach().to(device=U.device, dtype=torch.float32)
    V = factors_Vmat.detach().to(device=U.device, dtype=torch.float32)
    if gates is not None:
        s = s * gates.to(device=U.device, dtype=torch.float32)[..., : s.size(-1)]
    return torch.matmul(U * s.unsqueeze(-2), V.transpose(-1, -2))


def reconstruct_span_kv(
    span: SpanEntry,
    cos: torch.Tensor,
    sin: torch.Tensor,
    group: Optional[int] = None,
    gates_k: Optional[torch.Tensor] = None,
    gates_v: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    factors: Optional[SpanFactors] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a span's post-RoPE keys and values ([H_kv or 1, |S|, d_h] or [|S|, d_h] for a group).

    cos/sin are the RoPE tables at the span's positions; keys are rotated back to their original
    positions after the pre-RoPE low-rank reconstruction.
    """
    f = factors if factors is not None else span.factors
    if f is None:
        raise ValueError("Span has no factors to reconstruct")
    dev = device if device is not None else f.U_k.device

    def _pick(t: torch.Tensor, rank: Optional[int] = None) -> torch.Tensor:
        t = t if group is None else t[group]
        if rank is not None:
            t = t[..., :rank]
        return t.to(device=dev)

    rk = None if group is None else max(1, int(f.rank_k[group]))
    rv = None if group is None else max(1, int(f.rank_v[group]))
    k_pre = reconstruct_span(_pick(f.U_k, rk), _pick(f.s_k, rk), _pick(f.Vmat_k, rk), gates_k)
    v_hat = reconstruct_span(_pick(f.U_v, rv), _pick(f.s_v, rv), _pick(f.Vmat_v, rv), gates_v)
    return apply_rope(k_pre, cos.to(dev), sin.to(dev)), v_hat


def effective_rank(gates: torch.Tensor) -> torch.Tensor:
    """Effective rank r_S = sum_i m_i over the last dimension."""
    return gates.to(dtype=torch.float32).sum(dim=-1)
