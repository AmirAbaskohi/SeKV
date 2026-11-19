from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedModel

from sekv.backbone import FrozenBackbone, ModelSpec
from sekv.rope import apply_rope, rotate_half

__all__ = ["ArchAdapter", "build_arch_adapter", "rotate_half"]


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
        self._rotary = self._find_rotary()
        self._rotary_takes_position_ids = "position_ids" in inspect.signature(self._rotary.forward).parameters

    def _find_rotary(self):
        rotary = getattr(self.core, "rotary_emb", None)
        if rotary is None:
            rotary = getattr(self.core.layers[0].self_attn, "rotary_emb", None)
        if rotary is None:
            raise RuntimeError("Could not locate the rotary embedding module on the backbone")
        return rotary

    def rope_cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the backbone's RoPE (cos, sin), each [n, d_h] fp32, for absolute positions [n]."""
        inv_freq = getattr(self._rotary, "inv_freq", None)
        device = inv_freq.device if isinstance(inv_freq, torch.Tensor) else positions.device
        pos = positions.reshape(-1).to(device=device, dtype=torch.long)
        n = int(pos.numel())
        dummy = torch.zeros((1, 1, n, self.spec.head_dim), device=device, dtype=torch.float32)
        if self._rotary_takes_position_ids:
            cos, sin = self._rotary(dummy, pos.unsqueeze(0))
            cos = cos.reshape(n, -1)
            sin = sin.reshape(n, -1)
        else:
            seq_len = int(pos.max().item()) + 1 if n > 0 else 1
            cos, sin = self._rotary(dummy, seq_len=seq_len)
            cos = cos.reshape(-1, cos.size(-1)).index_select(0, pos)
            sin = sin.reshape(-1, sin.size(-1)).index_select(0, pos)
        return cos.to(dtype=torch.float32), sin.to(dtype=torch.float32)

    def apply_rope(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """Apply RoPE at one absolute position to per-head vectors x:[heads, d_h]."""
        pos = torch.tensor([int(position)], dtype=torch.long, device=x.device)
        cos, sin = self.rope_cos_sin(pos)
        return apply_rope(x, cos, sin)

    def apply_rope_q(self, layer: int, q: torch.Tensor, position: int) -> torch.Tensor:
        """Apply architecture RoPE to q at absolute decode position."""
        del layer
        return self.apply_rope(q, position)

    @abstractmethod
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[..., seq] -> [..., seq, hidden]"""

    @abstractmethod
    def project_qkv(self, layer: int, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return q:[H,d_h], k:[H_kv,d_h], v:[H_kv,d_h] before RoPE."""

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
