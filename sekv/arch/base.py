from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedModel

from sekv.backbone import FrozenBackbone, ModelSpec


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.size(-1) // 2]
    x2 = x[..., x.size(-1) // 2 :]
    return torch.cat([-x2, x1], dim=-1)


class ArchAdapter(ABC):
    """Uniform per-architecture access for SeKV explicit attention substitution."""

    def __init__(self, model: PreTrainedModel, spec: ModelSpec):
        self.model = model
        self.spec = spec
        if not hasattr(model, "model"):
            raise RuntimeError("Expected HF causal LM with .model core")
        self.core = model.model
        if not hasattr(self.core, "layers"):
            raise RuntimeError("Expected decoder stack at model.model.layers")

    @abstractmethod
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[..., seq] -> [..., seq, hidden]"""

    @abstractmethod
    def project_qkv(self, layer: int, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q:[H,d_h], k:[H_kv,d_h], v:[H_kv,d_h] before RoPE."""

    @abstractmethod
    def apply_rope_q(self, layer: int, q: torch.Tensor, position: int) -> torch.Tensor:
        """Apply architecture RoPE to q at absolute decode position."""

    @abstractmethod
    def input_layernorm(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def post_attention_layernorm(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def mlp(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        ...

    def project_attention_output(self, layer: int, attn_heads: torch.Tensor) -> torch.Tensor:
        if attn_heads.ndim != 2 or attn_heads.size(0) != self.spec.num_q_heads or attn_heads.size(1) != self.spec.head_dim:
            raise ValueError(
                f"attn_heads must be [{self.spec.num_q_heads}, {self.spec.head_dim}], got {tuple(attn_heads.shape)}"
            )
        layer_mod = self.core.layers[layer]
        return layer_mod.self_attn.o_proj(attn_heads.reshape(1, -1)).squeeze(0)

    def decoder_layer_forward(self, layer: int, hidden: torch.Tensor, attn_output: torch.Tensor) -> torch.Tensor:
        h = hidden + attn_output
        h = h + self.mlp(layer, self.post_attention_layernorm(layer, h))
        return h


from sekv.arch.llama import LlamaArchAdapter
from sekv.arch.mistral import MistralArchAdapter
from sekv.arch.qwen2 import Qwen2ArchAdapter


def build_arch_adapter(backbone: FrozenBackbone) -> ArchAdapter:
    name = backbone.spec.name
    if name in {"llama-3.2-3b", "llama-3-8b-base", "llama-3.1-8b"}:
        return LlamaArchAdapter(backbone.model, backbone.spec)
    if name == "mistral-7b":
        return MistralArchAdapter(backbone.model, backbone.spec)
    if name == "qwen2.5-14b":
        return Qwen2ArchAdapter(backbone.model, backbone.spec)
    raise KeyError(f"No ArchAdapter registered for model name: {name}")
