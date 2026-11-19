from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Optional

import torch

from sekv.arch import ArchAdapter, build_arch_adapter
from sekv.attention import BudgetTooSmallError, decode_step_layer, persistent_tokens
from sekv.backbone import FrozenBackbone, ModelSpec
from sekv.config import MemoryConfig, RoutingConfig, SeKVConfig, load_config
from sekv.memory import LayerMemory, SpanFactors, SpanMemory, build_span_memory, reconstruct_span_kv, roll_window
from sekv.modules import SeKVModules
from sekv.rope import RopeFn
from sekv.segmentation import memory_layout, segment_sequence


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    # None: no cap. Value <= 1: fraction of Full-KV (e.g. 0.1 = 10% GPU-resident KV budget).
    # Value > 1: absolute cap in tokens per KV head per layer.
    budget: float | int | None
    eos_token_id: int | None


@dataclass
class StepStats:
    """Per decoding step: peak per-layer resident KV (tokens per KV head), mean share of
    (span, head) decisions that zoom in, mean stored rank of reconstructed spans, host-to-device
    bytes, and the share of reconstructed spans reused from the previous step."""

    resident_kv_tokens: float
    expansion_rate: float
    effective_rank: float
    h2d_bytes: int
    reuse_rate: float


def resolve_budget(budget: float | int | None, context_len: int) -> float | None:
    """Convert a budget spec to tokens per KV head per layer."""
    if budget is None:
        return None
    value = float(budget)
    if value <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if value <= 1.0:
        return value * float(context_len)
    return value


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
    mismatches = [
        f"{name}: ckpt={ck.get(name)} runtime={rt.get(name)}"
        for name in ("name", "hf_id", "num_layers", "num_q_heads", "num_kv_heads", "head_dim", "group_size", "is_instruct")
        if ck.get(name) != rt.get(name)
    ]
    if mismatches:
        raise ValueError(f"Checkpoint ModelSpec mismatch with loaded backbone: {'; '.join(mismatches)}")


def _section(cfg_obj: Any, name: str, cls: type) -> Any:
    raw = getattr(cfg_obj, name, None) if not isinstance(cfg_obj, dict) else cfg_obj.get(name)
    if isinstance(raw, cls):
        return raw
    if isinstance(raw, dict):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})
    raise KeyError(f"Checkpoint config is missing the '{name}' section")


def load_sekv_checkpoint(path: str, backbone: FrozenBackbone) -> SeKVModules:
    """Load trainable params into SeKVModules and validate checkpoint spec against runtime spec."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint at {path} must be a dict payload")
    if "model_spec" not in ckpt or "trainable_parameters" not in ckpt or "config" not in ckpt:
        raise KeyError("Checkpoint must contain 'model_spec', 'config', and 'trainable_parameters'")

    _assert_spec_match(ckpt["model_spec"], backbone.spec)
    routing_cfg = _section(ckpt["config"], "routing", RoutingConfig)
    memory_raw = ckpt["config"]["memory"] if isinstance(ckpt["config"], dict) else ckpt["config"].memory
    memory_dict = asdict(memory_raw) if not isinstance(memory_raw, dict) else dict(memory_raw)
    memory_cfg = MemoryConfig(
        r_max=int(memory_dict["r_max"]),
        summary_dim=int(memory_dict["summary_dim"]),
        local_window=int(memory_dict.get("local_window", 512)),
        gate_eps=float(memory_dict.get("gate_eps", 0.05)),
    )

    modules = SeKVModules(spec=backbone.spec, routing=routing_cfg, memory=memory_cfg)
    trainable = ckpt["trainable_parameters"]
    state = modules.state_dict()
    missing = [name for name in state if name not in trainable]
    if missing:
        raise KeyError(f"Checkpoint missing parameter(s): {', '.join(missing)}")
    for name in state:
        tensor = trainable[name]
        if tuple(tensor.shape) != tuple(state[name].shape):
            raise ValueError(f"Shape mismatch for {name}: ckpt {tuple(tensor.shape)} vs model {tuple(state[name].shape)}")
        state[name] = tensor.to(dtype=state[name].dtype)

    modules.load_state_dict(state, strict=True)
    modules.eval()
    modules.to(device=backbone.model.device)
    for p in modules.parameters():
        p.requires_grad = False
    return modules


class ZoomCache:
    """GPU-side zoom-in state.

    Spans expanded at the previous step stay reconstructed on GPU and are reused; spans that stop
    being expanded are released, so the cache always matches the current budget accounting. New
    spans fetch only their KV group's factors (rank r_S) from pinned CPU memory on a side stream,
    and factors are cast to the model dtype after transfer.
    """

    def __init__(self, device: torch.device | str, rope_fn: RopeFn, dtype: torch.dtype):
        self.device = torch.device(device)
        self.rope_fn = rope_fn
        self.dtype = dtype
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self.stream = torch.cuda.Stream(device=self.device) if self.cuda else None
        self._recon: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._inflight: dict[tuple[int, int, int], tuple[SpanFactors, Optional[torch.cuda.Event]]] = {}
        self._pinned: set[int] = set()
        self.step_h2d_bytes = 0
        self.step_requests = 0
        self.step_reused = 0

    def start_step(self) -> None:
        self.step_h2d_bytes = 0
        self.step_requests = 0
        self.step_reused = 0

    def _pin(self, factors: SpanFactors) -> None:
        if not self.cuda or id(factors) in self._pinned or factors.U_k.device.type != "cpu":
            return
        for name in ("U_k", "s_k", "Vmat_k", "U_v", "s_v", "Vmat_v"):
            setattr(factors, name, getattr(factors, name).pin_memory())
        self._pinned.add(id(factors))

    def _group_factors(self, factors: SpanFactors, group: int) -> SpanFactors:
        rk, rv = max(1, factors.rank_k[group]), max(1, factors.rank_v[group])

        def _move(t: torch.Tensor, r: int) -> torch.Tensor:
            return t[group : group + 1, ..., :r].to(device=self.device, non_blocking=True).to(self.dtype)

        return SpanFactors(
            U_k=_move(factors.U_k, rk),
            s_k=_move(factors.s_k, rk),
            Vmat_k=_move(factors.Vmat_k, rk),
            U_v=_move(factors.U_v, rv),
            s_v=_move(factors.s_v, rv),
            Vmat_v=_move(factors.Vmat_v, rv),
            rank_k=[rk],
            rank_v=[rv],
            energy_k=factors.energy_k[group : group + 1],
            energy_v=factors.energy_v[group : group + 1],
            gated=factors.gated,
        )

    def _issue(self, layer: int, layer_mem: LayerMemory, span_idx: int, group: int) -> None:
        span = layer_mem.spans[span_idx]
        key = (layer, span.start, group)
        if key in self._inflight or span.factors is None:
            return
        self._pin(span.factors)
        self.step_h2d_bytes += span.factors.group_nbytes(group)
        if self.cuda:
            with torch.cuda.stream(self.stream):
                moved = self._group_factors(span.factors, group)
                event = torch.cuda.Event()
                event.record(self.stream)
        else:
            moved, event = self._group_factors(span.factors, group), None
        self._inflight[key] = (moved, event)

    def prepare(self, layer: int, layer_mem: LayerMemory, requests: list[tuple[int, int]]) -> None:
        wanted = {(layer, layer_mem.spans[s].start, g) for s, g in requests}
        for key in [k for k in self._recon if k[0] == layer and k not in wanted]:
            del self._recon[key]

        self.step_requests += len(requests)
        for span_idx, group in requests:
            key = (layer, layer_mem.spans[span_idx].start, group)
            if key in self._recon:
                self.step_reused += 1
                continue
            self._issue(layer, layer_mem, span_idx, group)

    def get(self, layer: int, layer_mem: LayerMemory, span_idx: int, group: int) -> tuple[torch.Tensor, torch.Tensor]:
        span = layer_mem.spans[span_idx]
        key = (layer, span.start, group)
        cached = self._recon.get(key)
        if cached is not None:
            return cached

        if key not in self._inflight:
            self._issue(layer, layer_mem, span_idx, group)
        factors, event = self._inflight.pop(key)
        if event is not None:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(event)
            for name in ("U_k", "s_k", "Vmat_k", "U_v", "s_v", "Vmat_v"):
                getattr(factors, name).record_stream(current)

        cos, sin = self.rope_fn(torch.arange(span.start, span.stop, device=self.device))
        k_hat, v_hat = reconstruct_span_kv(span, cos, sin, group=0, device=self.device, factors=factors)
        result = (k_hat.to(self.dtype), v_hat.to(self.dtype))
        self._recon[key] = result
        return result

    def finish_layer(self, layer: int, layer_mem: LayerMemory, requests: list[tuple[int, int]]) -> None:
        del layer_mem, requests
        for key in [k for k in self._inflight if k[0] == layer]:
            del self._inflight[key]

    def mean_rank(self, layer_mem: LayerMemory, requests: list[tuple[int, int]]) -> float:
        ranks = []
        for span_idx, group in requests:
            f = layer_mem.spans[span_idx].factors
            if f is not None:
                ranks.append(0.5 * (f.rank_k[group] + f.rank_v[group]))
        return sum(ranks) / len(ranks) if ranks else 0.0


# Backwards-compatible name for the async factor prefetcher.
SVDPrefetcher = ZoomCache


def _autocast(backbone: FrozenBackbone):
    if backbone.model.dtype == torch.bfloat16:
        return torch.autocast(device_type=backbone.model.device.type, dtype=torch.bfloat16)
    return nullcontext()


def sekv_forward_token(
    backbone: FrozenBackbone,
    adapter: ArchAdapter,
    modules: SeKVModules,
    memory: SpanMemory,
    token_id: torch.Tensor,
    position: int,
    cfg: SeKVConfig,
    budget: float | None,
    cache: ZoomCache,
) -> tuple[torch.Tensor, StepStats]:
    """One decode step: the token's own KV joins the local window, then every layer attends over
    anchors, window, coarse entries, and zoomed-in spans in a single softmax."""
    spec = backbone.spec
    device = backbone.model.device
    cache.start_step()

    resident: list[float] = []
    expand: list[float] = []
    ranks: list[float] = []

    with _autocast(backbone):
        x = adapter.embed_tokens(token_id.reshape(1).to(device=device, dtype=torch.long)).reshape(-1)
        for layer_idx in range(spec.num_layers):
            layer_mem = memory.layers[layer_idx]
            attn_in = adapter.input_layernorm(layer_idx, x)
            q, k, v = adapter.project_qkv(layer_idx, attn_in)
            q = adapter.apply_rope(q, position)
            k = adapter.apply_rope(k, position)
            layer_mem.append_window(k, v)

            out, info = decode_step_layer(
                q_layer=q,
                layer_mem=layer_mem,
                modules=modules,
                layer=layer_idx,
                spec=spec,
                cfg=cfg,
                budget=budget,
                training=False,
                reconstructor=cache,
                return_info=True,
            )
            resident.append(info.resident_tokens)
            expand.append(float(info.head_fine.float().mean().item()) if info.head_fine.numel() else 0.0)
            requests = [(info.comp_ids[s], g) for s, g in info.expanded.nonzero(as_tuple=False).tolist()]
            if requests:
                ranks.append(cache.mean_rank(layer_mem, requests))

            x = adapter.decoder_layer_forward(layer_idx, x, adapter.project_attention_output(layer_idx, out))

        logits = adapter.lm_head(adapter.final_norm(x)).reshape(-1)

    stats = StepStats(
        resident_kv_tokens=max(resident) if resident else 0.0,
        expansion_rate=sum(expand) / max(1, len(expand)),
        effective_rank=sum(ranks) / len(ranks) if ranks else 0.0,
        h2d_bytes=int(cache.step_h2d_bytes),
        reuse_rate=float(cache.step_reused) / float(cache.step_requests) if cache.step_requests else 0.0,
    )
    return logits.to(dtype=torch.float32), stats


@torch.no_grad()
def generate(
    backbone: FrozenBackbone,
    modules: SeKVModules,
    input_ids: torch.Tensor,
    cfg: SeKVConfig,
    gen_cfg: GenerationConfig,
    residency_callback: Optional[Callable[..., None]] = None,
) -> torch.Tensor:
    """Greedy generation with SeKV span memory under an optional GPU-resident KV budget.

    Raises BudgetTooSmallError when the resident state (anchors, window, summaries, coarse
    entries) cannot fit in the budget; such inputs fall outside SeKV's operating range.
    """
    if gen_cfg.max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {gen_cfg.max_new_tokens}")

    modules.eval()
    device = backbone.model.device
    spec = backbone.spec
    adapter = build_arch_adapter(backbone)
    prompt_ids = _ensure_1d_input_ids(input_ids).to(device=device)
    n = int(prompt_ids.numel())
    if n == 0:
        raise ValueError("prompt must contain at least one token")

    window = int(cfg.memory.local_window)
    roll = max(1, int(cfg.segmentation.l_gap))

    with _autocast(backbone):
        prefill = backbone.prefill(prompt_ids.unsqueeze(0))
    seg = segment_sequence(prefill.surprisal, cfg.segmentation)
    layout = memory_layout(seg, n - 1, window)
    # The last prompt token is re-encoded by the first decode step, which appends its own KV.
    memory = build_span_memory(prefill, seg, layout, cfg, spec, adapter.rope_cos_sin, modules=modules, window_stop=n - 1)
    surprisal = prefill.surprisal.clone()
    del prefill

    budget = resolve_budget(gen_cfg.budget, n)
    if budget is not None:
        peak_window = min(window + roll, memory.layers[0].window_len + max(1, gen_cfg.max_new_tokens))
        extra = peak_window - memory.layers[0].window_len
        floor = max(persistent_tokens(lm, spec, modules.summary_dim, extra_window=extra) for lm in memory.layers)
        if floor > budget:
            raise BudgetTooSmallError(
                f"Resident state needs {floor:.1f} tokens per KV head but the budget is {budget:.1f} "
                f"(context {n} tokens)"
            )

    cache = ZoomCache(device=device, rope_fn=adapter.rope_cos_sin, dtype=backbone.model.dtype)
    tokens = prompt_ids.clone()

    for step in range(gen_cfg.max_new_tokens):
        position = n - 1 + step
        logits, stats = sekv_forward_token(
            backbone, adapter, modules, memory, tokens[-1], position, cfg, budget, cache,
        )
        next_token = torch.argmax(logits, dim=-1).to(dtype=torch.long).view(1)
        token_surprisal = -torch.log_softmax(logits, dim=-1)[next_token].to("cpu", torch.float32)
        surprisal = torch.cat([surprisal, token_surprisal])
        tokens = torch.cat([tokens, next_token.to(device=tokens.device)], dim=0)

        if residency_callback is not None:
            residency_callback(
                stats.resident_kv_tokens, stats.expansion_rate, stats.effective_rank,
                stats.h2d_bytes, stats.reuse_rate,
            )
        if gen_cfg.eos_token_id is not None and int(next_token.item()) == int(gen_cfg.eos_token_id):
            break

        # Keep the local window bounded: its oldest tokens become a new span (non-evicting).
        if memory.layers[0].window_len >= window + roll:
            roll_window(memory, roll, surprisal, cfg, adapter.rope_cos_sin, modules, offload=True)

    return tokens


def generate_text(
    backbone: FrozenBackbone,
    modules: SeKVModules,
    prompt: str,
    cfg: SeKVConfig,
    gen_cfg: GenerationConfig,
    apply_chat_template: bool,
    residency_callback: Optional[Callable[..., None]] = None,
) -> str:
    """Tokenize prompt, run SeKV generation, and decode only new tokens."""
    tokenizer = backbone.tokenizer

    if backbone.spec.is_instruct and apply_chat_template and getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    input_ids = input_ids.to(device=backbone.model.device)
    full = generate(backbone, modules, input_ids=input_ids, cfg=cfg, gen_cfg=gen_cfg, residency_callback=residency_callback)

    prompt_len = int(_ensure_1d_input_ids(input_ids).numel())
    new_tokens = full[prompt_len:]
    if new_tokens.numel() == 0:
        return ""
    return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)


def parse_budget(raw: str | None) -> float | None:
    if raw is None or str(raw).strip().lower() in {"", "none", "null"}:
        return None
    return float(raw)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with SeKV")
    parser.add_argument("--config", type=str, default="sekv/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--budget",
        type=str,
        default="none",
        help="GPU-resident KV budget: 'none', a fraction of Full KV (e.g. 0.1), or tokens per KV head (e.g. 4096)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--apply-chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    overrides = {"model.name": args.model} if args.model else None
    cfg = load_config(args.config, overrides=overrides)

    backbone = FrozenBackbone(cfg.model)
    modules = load_sekv_checkpoint(args.checkpoint, backbone)
    gen_cfg = GenerationConfig(
        max_new_tokens=int(args.max_new_tokens),
        budget=parse_budget(args.budget),
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
