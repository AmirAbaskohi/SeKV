from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from typing import Any

import torch
from packaging.version import Version
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sekv.config import ModelConfig, SeKVConfig


_TRANSFORMERS_MIN = Version("4.44.0")
_TRANSFORMERS_MAX = Version("4.58.0")
_TRANSFORMERS_VERSION = Version(pkg_version("transformers"))
if not (_TRANSFORMERS_MIN <= _TRANSFORMERS_VERSION < _TRANSFORMERS_MAX):
    raise RuntimeError(
        "Unsupported transformers version for SeKV cache/RoPE conventions: "
        f"found {_TRANSFORMERS_VERSION}, expected >= {_TRANSFORMERS_MIN} and < {_TRANSFORMERS_MAX}. "
        "Install a compatible release, e.g. `pip install 'transformers>=4.44,<4.58'`."
    )


MODEL_REGISTRY: dict[str, str] = {
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3-8b-base": "meta-llama/Meta-Llama-3-8B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    group_size: int
    is_instruct: bool


@dataclass(frozen=True)
class PrefillOutputs:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    next_token_logits: torch.Tensor


def _require_int_attr(config: Any, attr: str, hf_id: str) -> int:
    value = getattr(config, attr, None)
    if value is None:
        raise ValueError(f"Model {hf_id} config is missing required attribute: {attr}")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"Model {hf_id} config attribute {attr} must be int, got {type(value).__name__}"
        )
    return value


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported values: {', '.join(mapping)}")
    return mapping[dtype]


def _resolve_model_input_device(model: PreTrainedModel) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for placement in device_map.values():
            if placement in {"disk", "meta"}:
                continue
            if isinstance(placement, int):
                return torch.device(f"cuda:{placement}")
            if isinstance(placement, str):
                return torch.device(placement)
    return model.device


def _extract_post_rope_kv(
    past_key_values: Any,
    num_layers: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return per-layer post-RoPE key/value tensors [B,H_kv,seq,d_h].

    Supports transformers cache objects (DynamicCache and close variants) and legacy
    tuple-of-tuples layouts. We keep post-RoPE keys as the single convention: decode
    queries are RoPE-rotated at true absolute position and compared directly against
    these keys across all supported architectures.
    """
    # Cache object with top-level key/value lists.
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        keys = list(getattr(past_key_values, "key_cache"))
        values = list(getattr(past_key_values, "value_cache"))
    # Cache object with per-layer containers.
    elif hasattr(past_key_values, "layers"):
        keys = []
        values = []
        for idx, layer_cache in enumerate(getattr(past_key_values, "layers")):
            if hasattr(layer_cache, "keys") and hasattr(layer_cache, "values"):
                k = getattr(layer_cache, "keys")
                v = getattr(layer_cache, "values")
            elif hasattr(layer_cache, "key") and hasattr(layer_cache, "value"):
                k = getattr(layer_cache, "key")
                v = getattr(layer_cache, "value")
            elif isinstance(layer_cache, (tuple, list)) and len(layer_cache) >= 2:
                k, v = layer_cache[0], layer_cache[1]
            else:
                raise TypeError(f"Unrecognized layer cache format at layer {idx}")
            keys.append(k)
            values.append(v)
    else:
        # Legacy layout or cache with conversion helper.
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()

        if not isinstance(past_key_values, (tuple, list)):
            raise TypeError(
                "Unrecognized past_key_values type returned by transformers "
                f"{_TRANSFORMERS_VERSION}: {type(past_key_values).__name__}."
            )

        keys = []
        values = []
        for idx, layer_cache in enumerate(past_key_values):
            if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
                raise TypeError(f"Unexpected legacy layer cache format at layer {idx}")
            keys.append(layer_cache[0])
            values.append(layer_cache[1])

    if len(keys) != num_layers or len(values) != num_layers:
        raise ValueError(
            f"Cache layer count mismatch: keys={len(keys)}, values={len(values)}, expected={num_layers}"
        )

    for idx, (k, v) in enumerate(zip(keys, values)):
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise TypeError(f"Layer {idx} cache entries must be tensors")

    return keys, values


class FrozenBackbone:
    def __init__(self, cfg: ModelConfig):
        if cfg.name not in MODEL_REGISTRY and not cfg.hf_id:
            raise KeyError(
                f"Unknown model.name '{cfg.name}'. Either use a registry key or provide model.hf_id."
            )

        hf_id = cfg.hf_id or MODEL_REGISTRY[cfg.name]
        hf_config = AutoConfig.from_pretrained(hf_id)

        num_layers = _require_int_attr(hf_config, "num_hidden_layers", hf_id)
        num_q_heads = _require_int_attr(hf_config, "num_attention_heads", hf_id)

        num_kv_heads_raw = getattr(hf_config, "num_key_value_heads", None)
        if num_kv_heads_raw is None:
            num_kv_heads = num_q_heads
        elif isinstance(num_kv_heads_raw, bool) or not isinstance(num_kv_heads_raw, int):
            raise TypeError(
                f"Model {hf_id} config attribute num_key_value_heads must be int when present"
            )
        else:
            num_kv_heads = num_kv_heads_raw

        hidden_size = _require_int_attr(hf_config, "hidden_size", hf_id)
        if hidden_size % num_q_heads != 0:
            raise ValueError(
                f"Invalid config for {hf_id}: hidden_size ({hidden_size}) is not divisible by "
                f"num_attention_heads ({num_q_heads})"
            )

        head_dim = hidden_size // num_q_heads
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                f"Invalid grouped-query setup for {hf_id}: num_attention_heads ({num_q_heads}) "
                f"must be divisible by num_key_value_heads ({num_kv_heads})"
            )
        group_size = num_q_heads // num_kv_heads

        if cfg.name == "mistral-7b":
            sliding_window = getattr(hf_config, "sliding_window", None)
            if sliding_window:
                raise ValueError(
                    "Mistral-7B-Instruct-v0.2 config has sliding_window enabled, but SeKV Phase 1 "
                    "assumes full attention spans. Please use a full-attention variant."
                )

        self._tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=_resolve_torch_dtype(cfg.dtype),
            device_map=cfg.device_map,
            attn_implementation=cfg.attn_implementation,
        )
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad = False

        self._spec = ModelSpec(
            name=cfg.name,
            hf_id=hf_id,
            num_layers=num_layers,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            group_size=group_size,
            is_instruct=(cfg.name != "llama-3-8b-base"),
        )
        self._num_layers = num_layers

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def q_for_query_head(self, h: int) -> int:
        if h < 0 or h >= self._spec.num_q_heads:
            raise IndexError(f"Query head index out of range: {h}")
        return h

    def kv_group_for_query_head(self, h: int) -> int:
        if h < 0 or h >= self._spec.num_q_heads:
            raise IndexError(f"Query head index out of range: {h}")
        return h // self._spec.group_size

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> PrefillOutputs:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, seq_len], got {tuple(input_ids.shape)}"
            )
        if input_ids.size(0) != 1:
            raise ValueError(
                f"prefill currently expects batch size 1, got batch size {input_ids.size(0)}"
            )

        model_input_device = _resolve_model_input_device(self._model)
        input_ids = input_ids.to(device=model_input_device)

        outputs = self._model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        # We deliberately read post-RoPE keys from past_key_values so reconstructed tokens
        # re-enter attention directly in query space without re-applying rotary embedding.
        # Value states are position-free and are consumed as-is.
        # For Qwen2.5, QKV bias is already baked into these K/V states, so downstream
        # summary/SVD operations should use them directly without bias stripping.
        key_cache, value_cache = _extract_post_rope_kv(outputs.past_key_values, self._num_layers)

        target_device = outputs.logits.device
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        for layer_idx, (key_states, value_states) in enumerate(zip(key_cache, value_cache)):
            if key_states.ndim != 4 or value_states.ndim != 4:
                raise ValueError(
                    f"Layer {layer_idx} cache tensors must be rank-4, got "
                    f"K{tuple(key_states.shape)} and V{tuple(value_states.shape)}"
                )
            if key_states.size(0) != 1 or value_states.size(0) != 1:
                raise ValueError(
                    f"Layer {layer_idx} cache batch dimension must be 1, got "
                    f"K batch={key_states.size(0)}, V batch={value_states.size(0)}"
                )

            keys.append(key_states[0].to(device=target_device).contiguous())
            values.append(value_states[0].to(device=target_device).contiguous())

        return PrefillOutputs(
            keys=tuple(keys),
            values=tuple(values),
            next_token_logits=outputs.logits[0].to(device=target_device).contiguous(),
        )


def build_backbone(cfg: SeKVConfig) -> FrozenBackbone:
    return FrozenBackbone(cfg.model)
