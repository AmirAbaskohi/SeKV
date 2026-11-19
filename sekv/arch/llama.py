from __future__ import annotations

import torch
from transformers import PreTrainedModel

from sekv.arch.base import ArchAdapter
from sekv.backbone import ModelSpec


class LlamaArchAdapter(ArchAdapter):
    def __init__(self, model: PreTrainedModel, spec: ModelSpec):
        super().__init__(model, spec)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.core.embed_tokens(input_ids)

    def _split_heads(self, x: torch.Tensor, heads: int) -> torch.Tensor:
        return x.view(heads, self.spec.head_dim)

    def project_qkv(self, layer: int, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_mod = self.core.layers[layer]
        h = hidden.unsqueeze(0)
        q = layer_mod.self_attn.q_proj(h).squeeze(0)
        k = layer_mod.self_attn.k_proj(h).squeeze(0)
        v = layer_mod.self_attn.v_proj(h).squeeze(0)
        return (
            self._split_heads(q, self.spec.num_q_heads),
            self._split_heads(k, self.spec.num_kv_heads),
            self._split_heads(v, self.spec.num_kv_heads),
        )

    def input_layernorm(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.core.layers[layer].input_layernorm(hidden.unsqueeze(0)).squeeze(0)

    def post_attention_layernorm(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.core.layers[layer].post_attention_layernorm(hidden.unsqueeze(0)).squeeze(0)

    def mlp(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.core.layers[layer].mlp(hidden.unsqueeze(0)).squeeze(0)

    def final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.core.norm(hidden.unsqueeze(0)).squeeze(0)

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(hidden.unsqueeze(0)).squeeze(0)
