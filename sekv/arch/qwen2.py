from __future__ import annotations

import torch
from transformers import PreTrainedModel

from sekv.arch.llama import LlamaArchAdapter
from sekv.backbone import ModelSpec


class Qwen2ArchAdapter(LlamaArchAdapter):
    def __init__(self, model: PreTrainedModel, spec: ModelSpec):
        super().__init__(model, spec)

    def project_qkv(self, layer: int, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Qwen2 q/k/v projections include bias; use the model modules directly so bias is preserved.
        # Stored post-RoPE keys from prefill already include this bias, keeping SVD/summaries consistent.
        layer_mod = self.core.layers[layer]
        h = hidden.unsqueeze(0)
        q = layer_mod.self_attn.q_proj(h).squeeze(0)
        k = layer_mod.self_attn.k_proj(h).squeeze(0)
        v = layer_mod.self_attn.v_proj(h).squeeze(0)
        return (
            q.view(self.spec.num_q_heads, self.spec.head_dim),
            k.view(self.spec.num_kv_heads, self.spec.head_dim),
            v.view(self.spec.num_kv_heads, self.spec.head_dim),
        )
