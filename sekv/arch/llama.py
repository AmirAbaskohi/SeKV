from __future__ import annotations

import torch
from transformers import PreTrainedModel

from sekv.arch.base import ArchAdapter, rotate_half
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

    def apply_rope_q(self, layer: int, q: torch.Tensor, position: int) -> torch.Tensor:
        layer_mod = self.core.layers[layer]
        if not hasattr(layer_mod.self_attn, "rotary_emb"):
            return q

        # Convention match: decode query RoPE here must match the post-RoPE key convention stored from prefill.
        q4 = q.unsqueeze(0).unsqueeze(2)
        pos_ids = torch.tensor([[int(position)]], device=q.device, dtype=torch.long)
        rotary = layer_mod.self_attn.rotary_emb

        try:
            cos, sin = rotary(q4, pos_ids)
        except TypeError:
            cos, sin = rotary(q4, position_ids=pos_ids)

        while cos.ndim < q4.ndim:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_rot = (q4 * cos) + (rotate_half(q4) * sin)
        return q_rot.squeeze(0).squeeze(1)

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
