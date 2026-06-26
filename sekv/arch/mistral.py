from __future__ import annotations

from transformers import PreTrainedModel

from sekv.arch.llama import LlamaArchAdapter
from sekv.backbone import ModelSpec


class MistralArchAdapter(LlamaArchAdapter):
    """Mistral decoder shares the same explicit hook points as Llama for SeKV substitution."""

    def __init__(self, model: PreTrainedModel, spec: ModelSpec):
        super().__init__(model, spec)
