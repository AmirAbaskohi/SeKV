from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch

from sekv.arch import ArchAdapter, build_arch_adapter
from sekv.attention import budget_select, decode_step_layer
from sekv.backbone import FrozenBackbone, ModelSpec
from sekv.config import MemoryConfig, RoutingConfig, SeKVConfig, load_config
from sekv.memory import LayerMemory, SpanFactors, SpanMemory, build_span_memory
from sekv.modules import SeKVModules
from sekv.segmentation import segment_prefill


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    budget: int | None
    eos_token_id: int | None


def _ensure_1d_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim == 2:
        if input_ids.size(0) != 1:
            raise ValueError(f"input_ids must be [seq] or [1,seq], got {tuple(input_ids.shape)}")
        input_ids = input_ids[0]
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be [seq] or [1,seq], got {tuple(input_ids.shape)}")
    return input_ids.to(dtype=torch.long)


def _spec_as_dict(spec: ModelSpec | dict[str, Any]) -> dict[str, Any]:
    if isinstance(spec, dict):
        return dict(spec)
    return asdict(spec)


def _assert_spec_match(checkpoint_spec: ModelSpec | dict[str, Any], runtime_spec: ModelSpec) -> None:
    ck = _spec_as_dict(checkpoint_spec)
    rt = asdict(runtime_spec)

    fields = [
        "name",
        "hf_id",
        "num_layers",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "group_size",
        "is_instruct",
    ]
    mismatches: list[str] = []
    for field in fields:
        if ck.get(field) != rt.get(field):
            mismatches.append(f"{field}: ckpt={ck.get(field)} runtime={rt.get(field)}")

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"Checkpoint ModelSpec mismatch with loaded backbone: {mismatch_text}")


def _span_factors_to_device(factors: SpanFactors, device: torch.device, non_blocking: bool) -> SpanFactors:
    return SpanFactors(
        U_k=factors.U_k.to(device=device, dtype=torch.float32, non_blocking=non_blocking).contiguous(),
        s_k=factors.s_k.to(device=device, dtype=torch.float32, non_blocking=non_blocking).contiguous(),
        Vmat_k=factors.Vmat_k.to(device=device, dtype=torch.float32, non_blocking=non_blocking).contiguous(),
        U_v=factors.U_v.to(device=device, dtype=torch.float32, non_blocking=non_blocking).contiguous(),
        s_v=factors.s_v.to(device=device, dtype=torch.float32, non_blocking=non_blocking).contiguous(),
        Vmat_v=factors.Vmat_v.to(device=device, dtype=torch.float32, non_blocking=non_blocking).contiguous(),
        valid_rank=int(factors.valid_rank),
    )


def load_sekv_checkpoint(path: str, backbone: FrozenBackbone) -> SeKVModules:
    """Load trainable params into SeKVModules and validate checkpoint spec against runtime spec."""
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} must be a dict payload")

    if "model_spec" not in ckpt or "trainable_parameters" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_spec' and 'trainable_parameters'")

    _assert_spec_match(ckpt["model_spec"], backbone.spec)

    ckpt_cfg = ckpt.get("config")
    if hasattr(ckpt_cfg, "routing") and hasattr(ckpt_cfg, "memory"):
        routing_cfg = ckpt_cfg.routing
        memory_cfg = ckpt_cfg.memory
    elif isinstance(ckpt_cfg, dict) and "routing" in ckpt_cfg and "memory" in ckpt_cfg:
        routing_raw = ckpt_cfg["routing"]
        memory_raw = ckpt_cfg["memory"]
        if isinstance(routing_raw, RoutingConfig):
            routing_cfg = routing_raw
        elif isinstance(routing_raw, dict):
            routing_cfg = RoutingConfig(**routing_raw)
        else:
            raise TypeError("Checkpoint routing config must be RoutingConfig or dict")

        if isinstance(memory_raw, MemoryConfig):
            memory_cfg = memory_raw
        elif isinstance(memory_raw, dict):
            memory_cfg = MemoryConfig(**memory_raw)
        else:
            raise TypeError("Checkpoint memory config must be MemoryConfig or dict")
    else:
        raise KeyError(
            "Checkpoint must include full config with 'routing' and 'memory' to rebuild SeKVModules"
        )

    modules = SeKVModules(spec=backbone.spec, routing=routing_cfg, memory=memory_cfg)

    # Rebuild modules with runtime config if checkpoint config isn't a dataclass tree.
    if not isinstance(modules, SeKVModules):
        raise RuntimeError("Failed to build SeKVModules")

    trainable = ckpt["trainable_parameters"]
    if not isinstance(trainable, dict):
        raise TypeError("Checkpoint 'trainable_parameters' must be a dict[str, Tensor]")

    state = modules.state_dict()
    missing: list[str] = []
    for name in state:
        if name not in trainable:
            missing.append(name)
            continue
        tensor = trainable[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Checkpoint parameter {name} must be a Tensor")
        if tuple(tensor.shape) != tuple(state[name].shape):
            raise ValueError(
                f"Shape mismatch for {name}: ckpt {tuple(tensor.shape)} vs model {tuple(state[name].shape)}"
            )
        state[name] = tensor.to(dtype=state[name].dtype)

    if missing:
        raise KeyError(f"Checkpoint missing parameter(s): {', '.join(missing)}")

    modules.load_state_dict(state, strict=True)
    modules.eval()
    modules.to(device=backbone.model.device)
    for p in modules.parameters():
        p.requires_grad = False

    return modules


class SVDPrefetcher:
    """Async CPU->GPU prefetch for span factors with bounded LRU cache."""

    def __init__(self, device, max_cached_spans: int):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.max_cached_spans = int(max_cached_spans)
        if self.max_cached_spans <= 0:
            raise ValueError(f"max_cached_spans must be > 0, got {max_cached_spans}")

        self.cuda_enabled = self.device.type == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream(device=self.device) if self.cuda_enabled else None

        self._pinned_cpu: dict[tuple[int, int], SpanFactors] = {}
        self._gpu_cache: OrderedDict[tuple[int, int], SpanFactors] = OrderedDict()
        self._inflight: dict[tuple[int, int], tuple[SpanFactors, torch.cuda.Event]] = {}

    def _pin_cpu_factors(self, key: tuple[int, int], factors: SpanFactors) -> SpanFactors:
        if key in self._pinned_cpu:
            return self._pinned_cpu[key]

        if factors.U_k.device.type != "cpu":
            self._pinned_cpu[key] = factors
            return factors

        pinned = SpanFactors(
            U_k=factors.U_k.pin_memory(),
            s_k=factors.s_k.pin_memory(),
            Vmat_k=factors.Vmat_k.pin_memory(),
            U_v=factors.U_v.pin_memory(),
            s_v=factors.s_v.pin_memory(),
            Vmat_v=factors.Vmat_v.pin_memory(),
            valid_rank=int(factors.valid_rank),
        )
        self._pinned_cpu[key] = pinned
        return pinned

    def request(self, layer: int, span_id: int, factors: SpanFactors) -> None:
        key = (int(layer), int(span_id))
        if key in self._gpu_cache or key in self._inflight:
            return

        if not self.cuda_enabled:
            return

        src = self._pin_cpu_factors(key, factors)
        with torch.cuda.stream(self.stream):
            gpu = _span_factors_to_device(src, device=self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.stream)
        self._inflight[key] = (gpu, event)

    def get(self, layer: int, span_id: int) -> SpanFactors | None:
        key = (int(layer), int(span_id))

        cached = self._gpu_cache.get(key)
        if cached is not None:
            self._gpu_cache.move_to_end(key)
            return cached

        inflight = self._inflight.get(key)
        if inflight is None:
            return None

        factors, event = inflight
        if self.cuda_enabled and not event.query():
            return None

        if self.cuda_enabled:
            event.synchronize()
        self._inflight.pop(key, None)
        self._gpu_cache[key] = factors
        self._gpu_cache.move_to_end(key)
        self.evict_if_needed()
        return factors

    def evict_if_needed(self) -> None:
        while len(self._gpu_cache) > self.max_cached_spans:
            self._gpu_cache.popitem(last=False)


def _compressed_span_indices(layer_mem: LayerMemory) -> list[int]:
    return [i for i, span in enumerate(layer_mem.spans) if not span.full_resolution]


def _assert_budget_residency(
    layer_mem: LayerMemory,
    compressed_span_ids: list[int],
    per_group_sets: list[set[int]],
    budget: int | None,
) -> None:
    if budget is None:
        return

    num_anchors = int(layer_mem.anchor_k.size(1))
    num_full_tokens = sum(int(span.length) for span in layer_mem.spans if span.full_resolution)
    num_compressed = len(compressed_span_ids)
    persistent = num_anchors + num_full_tokens + num_compressed

    if persistent > budget:
        raise ValueError(
            f"Persistent KV residency ({persistent}) exceeds budget ({budget})"
        )

    for group_idx, comp_set in enumerate(per_group_sets):
        extra = 0
        for comp_idx in comp_set:
            span_id = compressed_span_ids[comp_idx]
            extra += max(0, int(layer_mem.spans[span_id].length) - 1)
        total = persistent + extra
        if total > budget:
            raise RuntimeError(
                f"Budget violated at group {group_idx}: residency={total}, budget={budget}"
            )


def _per_group_reconstruction_sets(
    alpha_tilde: torch.Tensor,
    tau_layer: torch.Tensor,
    span_lengths: torch.Tensor,
    layer_mem: LayerMemory,
    spec: ModelSpec,
    budget: int | None,
) -> list[set[int]]:
    # alpha_tilde: [num_compressed_spans, H]
    groups: list[set[int]] = [set() for _ in range(spec.num_kv_heads)]
    num_compressed = int(alpha_tilde.size(0))
    if num_compressed == 0:
        return groups

    for group in range(spec.num_kv_heads):
        h0 = group * spec.group_size
        h1 = h0 + spec.group_size

        if budget is None:
            selected = alpha_tilde[:, h0:h1] > tau_layer[h0:h1].unsqueeze(0)
            for idx in range(num_compressed):
                if torch.any(selected[idx]):
                    groups[group].add(idx)
        else:
            select_matrix: list[torch.Tensor] = []
            for h in range(h0, h1):
                sel_h = budget_select(
                    alpha_tilde_head=alpha_tilde[:, h],
                    tau_head=tau_layer[h],
                    span_lengths=span_lengths,
                    layer_mem=layer_mem,
                    budget=budget,
                )
                select_matrix.append(sel_h)
            selected = torch.stack(select_matrix, dim=1)
            for idx in range(num_compressed):
                if torch.any(selected[idx]):
                    groups[group].add(idx)

    return groups


def _ensure_factors_for_layer(
    layer_mem: LayerMemory,
    compressed_span_ids: list[int],
    per_group_sets: list[set[int]],
    prefetcher: SVDPrefetcher | None,
    layer: int,
    device: torch.device,
) -> None:
    if not compressed_span_ids:
        return

    # Request prefetch for routed spans first.
    for group, comp_set in enumerate(per_group_sets):
        del group
        for comp_idx in comp_set:
            span_id = compressed_span_ids[comp_idx]
            span = layer_mem.spans[span_id]
            if span.factors is None:
                continue
            if prefetcher is not None:
                prefetcher.request(layer=layer, span_id=span_id, factors=span.factors)

    # Ensure factors are resident before reconstruction. Async and sync paths must be numerically
    # identical; fallback blocking copies preserve correctness when prefetch misses timing.
    for comp_idx in set().union(*per_group_sets) if per_group_sets else set():
        span_id = compressed_span_ids[comp_idx]
        span = layer_mem.spans[span_id]
        if span.factors is None:
            continue

        cached = prefetcher.get(layer=layer, span_id=span_id) if prefetcher is not None else None
        if cached is not None:
            span.factors = cached
            continue

        span.factors = _span_factors_to_device(span.factors, device=device, non_blocking=False)


def _request_next_layer_prefetch(
    memory: SpanMemory,
    current_layer: int,
    compressed_span_ids: list[int],
    per_group_sets: list[set[int]],
    prefetcher: SVDPrefetcher | None,
) -> None:
    if prefetcher is None:
        return

    next_layer = current_layer + 1
    if next_layer >= len(memory.layers):
        return

    next_layer_mem = memory.layers[next_layer]
    for comp_idx in set().union(*per_group_sets) if per_group_sets else set():
        span_id = compressed_span_ids[comp_idx]
        span = next_layer_mem.spans[span_id]
        if span.factors is not None:
            prefetcher.request(layer=next_layer, span_id=span_id, factors=span.factors)


def _layer_residency_stats(
    layer_mem: LayerMemory,
    compressed_span_ids: list[int],
    per_group_sets: list[set[int]],
    modules: SeKVModules,
    memory: SpanMemory,
    layer: int,
    device: torch.device,
) -> tuple[int, float, float]:
    num_anchors = int(layer_mem.anchor_k.size(1))
    num_full_tokens = sum(int(span.length) for span in layer_mem.spans if span.full_resolution)
    num_compressed = len(compressed_span_ids)
    persistent = num_anchors + num_full_tokens + num_compressed

    if num_compressed == 0:
        return persistent, 0.0, 0.0

    per_group_residency: list[int] = []
    per_group_expand_rate: list[float] = []
    eff_rank_values: list[float] = []

    for group, comp_set in enumerate(per_group_sets):
        extra = 0
        for comp_idx in comp_set:
            span_id = compressed_span_ids[comp_idx]
            extra += max(0, int(layer_mem.spans[span_id].length) - 1)
        per_group_residency.append(persistent + extra)
        per_group_expand_rate.append(float(len(comp_set)) / float(max(1, num_compressed)))

        for comp_idx in comp_set:
            span_id = compressed_span_ids[comp_idx]
            factors = memory.layers[layer].spans[span_id].factors
            if factors is None:
                continue

            rank = int(factors.valid_rank)
            r_max = modules.g_phi.r_max
            valid_rank = torch.tensor([rank], device=device, dtype=torch.long)

            ctx = layer_mem.spans[span_id].k_bar[group].to(device=device, dtype=torch.float32)
            ctx = ctx / ctx.norm(p=2).clamp_min(1.0e-8)
            ctx = ctx.unsqueeze(0)

            svk = factors.s_k[group].to(device=device, dtype=torch.float32)
            svv = factors.s_v[group].to(device=device, dtype=torch.float32)

            svk_pad = torch.zeros((1, r_max), device=device, dtype=torch.float32)
            svv_pad = torch.zeros((1, r_max), device=device, dtype=torch.float32)
            svk_pad[:, :rank] = svk
            svv_pad[:, :rank] = svv

            gk = modules.g_phi(singular_values=svk_pad, context=ctx, valid_rank=valid_rank)
            gv = modules.g_phi(singular_values=svv_pad, context=ctx, valid_rank=valid_rank)
            eff_rank_values.append(float(gk.sum().item()))
            eff_rank_values.append(float(gv.sum().item()))

    residency = max(per_group_residency) if per_group_residency else persistent
    mean_expand = (
        sum(per_group_expand_rate) / float(len(per_group_expand_rate))
        if per_group_expand_rate
        else 0.0
    )
    mean_rank = (
        sum(eff_rank_values) / float(len(eff_rank_values))
        if eff_rank_values
        else 0.0
    )
    return residency, mean_expand, mean_rank


def sekv_forward_token(
    backbone: FrozenBackbone,
    adapter: ArchAdapter,
    modules: SeKVModules,
    memory: SpanMemory,
    hidden_or_ids,
    position: int,
    cfg: SeKVConfig,
    budget: int | None,
    prefetcher: SVDPrefetcher | None,
    residency_callback: Callable[[int, float, float], None] | None = None,
) -> torch.Tensor:
    """One decode step with frozen per-layer stack and SeKV mixed-resolution attention."""
    spec = backbone.spec
    device = backbone.model.device
    autocast_enabled = backbone.model.dtype == torch.bfloat16

    if isinstance(hidden_or_ids, torch.Tensor) and hidden_or_ids.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        token_id = hidden_or_ids
        if token_id.ndim == 0:
            token_id = token_id.unsqueeze(0)
        if token_id.ndim > 1:
            token_id = token_id.reshape(-1)
        token_id = token_id.to(device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            x = adapter.embed_tokens(token_id.unsqueeze(0)).squeeze(0).squeeze(0)
    elif isinstance(hidden_or_ids, torch.Tensor):
        x = hidden_or_ids.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("hidden_or_ids must be a Tensor token id or hidden state")

    tau_layer_all = modules.tau.to(device=device, dtype=torch.float32)
    layer_residencies: list[int] = []
    layer_expand_rates: list[float] = []
    layer_eff_ranks: list[float] = []

    for layer_idx in range(spec.num_layers):
        layer_mem = memory.layers[layer_idx]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            attn_in = adapter.input_layernorm(layer_idx, x)
        q_heads, _k_heads, _v_heads = adapter.project_qkv(layer_idx, attn_in)
        q_heads = adapter.apply_rope_q(layer_idx, q_heads.to(dtype=torch.float32), position=position)

        compressed_span_ids = _compressed_span_indices(layer_mem)
        if compressed_span_ids:
            k_bar = torch.stack([layer_mem.spans[sid].k_bar for sid in compressed_span_ids], dim=0)
            k_bar = k_bar.to(device=device, dtype=torch.float32)
            span_lengths = torch.tensor(
                [layer_mem.spans[sid].length for sid in compressed_span_ids],
                device=device,
                dtype=torch.float32,
            )

            K_bar = modules.routing_summaries(layer_idx, k_bar)
            q_proj_route = modules.project_query(layer_idx, q_heads)
            alpha_tilde = modules.relevance_gate(q_proj_route, K_bar, span_lengths)

            per_group_sets = _per_group_reconstruction_sets(
                alpha_tilde=alpha_tilde,
                tau_layer=tau_layer_all[layer_idx],
                span_lengths=span_lengths,
                layer_mem=layer_mem,
                spec=spec,
                budget=budget,
            )

            _assert_budget_residency(
                layer_mem=layer_mem,
                compressed_span_ids=compressed_span_ids,
                per_group_sets=per_group_sets,
                budget=budget,
            )

            _request_next_layer_prefetch(
                memory=memory,
                current_layer=layer_idx,
                compressed_span_ids=compressed_span_ids,
                per_group_sets=per_group_sets,
                prefetcher=prefetcher,
            )

            _ensure_factors_for_layer(
                layer_mem=layer_mem,
                compressed_span_ids=compressed_span_ids,
                per_group_sets=per_group_sets,
                prefetcher=prefetcher,
                layer=layer_idx,
                device=device,
            )

            layer_residency, layer_expand, layer_rank = _layer_residency_stats(
                layer_mem=layer_mem,
                compressed_span_ids=compressed_span_ids,
                per_group_sets=per_group_sets,
                modules=modules,
                memory=memory,
                layer=layer_idx,
                device=device,
            )
            layer_residencies.append(layer_residency)
            layer_expand_rates.append(layer_expand)
            layer_eff_ranks.append(layer_rank)
        else:
            num_anchors = int(layer_mem.anchor_k.size(1))
            num_full_tokens = sum(int(span.length) for span in layer_mem.spans if span.full_resolution)
            layer_residencies.append(num_anchors + num_full_tokens)
            layer_expand_rates.append(0.0)
            layer_eff_ranks.append(0.0)

        attn_heads = decode_step_layer(
            q_layer=q_heads,
            layer_mem=layer_mem,
            modules=modules,
            layer=layer_idx,
            spec=spec,
            cfg=cfg,
            budget=budget,
            training=False,
            ste_temperature=float(cfg.routing.ste_temperature),
        )

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            attn_out = adapter.project_attention_output(layer_idx, attn_heads)
        x = adapter.decoder_layer_forward(layer_idx, x, attn_out)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        x = adapter.final_norm(x)
        logits = adapter.lm_head(x).squeeze(0)

    if residency_callback is not None:
        residency_callback(
            int(max(layer_residencies) if layer_residencies else 0),
            float(sum(layer_expand_rates) / float(max(1, len(layer_expand_rates)))),
            float(sum(layer_eff_ranks) / float(max(1, len(layer_eff_ranks)))),
        )
    return logits.to(dtype=torch.float32)


@torch.no_grad()
def generate(
    backbone: FrozenBackbone,
    modules: SeKVModules,
    input_ids: torch.Tensor,
    cfg: SeKVConfig,
    gen_cfg: GenerationConfig,
    residency_callback: Callable[[int, float, float], None] | None = None,
) -> torch.Tensor:
    """Greedy generation with fixed prompt span memory and optional budgeted expansion."""
    if gen_cfg.max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {gen_cfg.max_new_tokens}")

    modules.eval()
    for p in modules.parameters():
        p.requires_grad = False

    prompt_ids = _ensure_1d_input_ids(input_ids).to(device=backbone.model.device)

    autocast_enabled = backbone.model.dtype == torch.bfloat16
    with torch.autocast(
        device_type=backbone.model.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        prefill = backbone.prefill(prompt_ids.unsqueeze(0))

    seg = segment_prefill(prefill=prefill, input_ids=prompt_ids, cfg=cfg.segmentation)
    memory = build_span_memory(prefill=prefill, seg=seg, cfg=cfg, spec=backbone.spec)
    adapter = build_arch_adapter(backbone)

    prefetcher: SVDPrefetcher | None
    if backbone.model.device.type == "cuda" and torch.cuda.is_available():
        prefetcher = SVDPrefetcher(device=backbone.model.device, max_cached_spans=256)
    else:
        prefetcher = None

    tokens = prompt_ids.clone()

    for _ in range(gen_cfg.max_new_tokens):
        current_token = tokens[-1]
        position = int(tokens.numel() - 1)

        logits = sekv_forward_token(
            backbone=backbone,
            adapter=adapter,
            modules=modules,
            memory=memory,
            hidden_or_ids=current_token,
            position=position,
            cfg=cfg,
            budget=gen_cfg.budget,
            prefetcher=prefetcher,
            residency_callback=residency_callback,
        )

        next_token = torch.argmax(logits, dim=-1).to(dtype=torch.long).view(1)
        tokens = torch.cat([tokens, next_token.to(device=tokens.device)], dim=0)

        if gen_cfg.eos_token_id is not None and int(next_token.item()) == int(gen_cfg.eos_token_id):
            break

    return tokens


def generate_text(
    backbone: FrozenBackbone,
    modules: SeKVModules,
    prompt: str,
    cfg: SeKVConfig,
    gen_cfg: GenerationConfig,
    apply_chat_template: bool,
    residency_callback: Callable[[int, float, float], None] | None = None,
) -> str:
    """Tokenize prompt, run SeKV generation, and decode only new tokens."""
    tokenizer = backbone.tokenizer

    if backbone.spec.is_instruct and apply_chat_template:
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        else:
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    input_ids = input_ids.to(device=backbone.model.device)
    full = generate(
        backbone,
        modules,
        input_ids=input_ids,
        cfg=cfg,
        gen_cfg=gen_cfg,
        residency_callback=residency_callback,
    )

    prompt_len = int(_ensure_1d_input_ids(input_ids).numel())
    new_tokens = full[prompt_len:]
    if new_tokens.numel() == 0:
        return ""

    return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with SeKV")
    parser.add_argument("--config", type=str, default="sekv/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--budget", type=str, default="none")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--apply-chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {"model.name": args.model} if args.model else None
    cfg = load_config(args.config, overrides=overrides)

    budget = None if str(args.budget).lower() in {"none", "null"} else int(args.budget)

    backbone = FrozenBackbone(cfg.model)
    modules = load_sekv_checkpoint(args.checkpoint, backbone)
    gen_cfg = GenerationConfig(
        max_new_tokens=int(args.max_new_tokens),
        budget=budget,
        eos_token_id=backbone.tokenizer.eos_token_id,
    )

    text = generate_text(
        backbone=backbone,
        modules=modules,
        prompt=args.prompt,
        cfg=cfg,
        gen_cfg=gen_cfg,
        apply_chat_template=bool(args.apply_chat_template),
    )
    print(text)


if __name__ == "__main__":
    main()
