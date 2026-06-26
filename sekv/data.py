from __future__ import annotations

import random
from typing import Iterable

import torch
from torch.utils.data import DataLoader, IterableDataset

from sekv.config import SeKVConfig


class RedPajamaLongDocuments(IterableDataset):
    """Streaming RedPajama iterator over arxiv/book/github with curriculum-adjustable length."""

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        subsets: tuple[str, ...] = ("arxiv", "book", "github"),
        seed: int = 0,
        min_doc_tokens: int | None = None,
    ):
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")

        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.subsets = tuple(subsets)
        self.seed = int(seed)
        self.min_doc_tokens = int(min_doc_tokens) if min_doc_tokens is not None else None

    def set_seq_len(self, seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        self.seq_len = int(seq_len)

    def _stream_for_subset(self, subset: str):
        from datasets import load_dataset

        return iter(
            load_dataset(
                "togethercomputer/RedPajama-Data-1T",
                name=subset,
                split="train",
                streaming=True,
            )
        )

    def _extract_text(self, sample: dict) -> str | None:
        for key in ("text", "raw_content", "content"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def __iter__(self) -> Iterable[torch.Tensor]:
        rng = random.Random(self.seed)
        streams = {subset: self._stream_for_subset(subset) for subset in self.subsets}
        subset_cycle = list(self.subsets)

        while True:
            rng.shuffle(subset_cycle)
            for subset in subset_cycle:
                stream = streams[subset]
                try:
                    sample = next(stream)
                except StopIteration:
                    streams[subset] = self._stream_for_subset(subset)
                    sample = next(streams[subset])

                text = self._extract_text(sample)
                if text is None:
                    continue

                encoded = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                )
                token_ids = encoded.get("input_ids", [])
                if not isinstance(token_ids, list):
                    continue

                min_tokens = self.min_doc_tokens if self.min_doc_tokens is not None else self.seq_len
                if len(token_ids) < min_tokens or len(token_ids) < self.seq_len:
                    continue

                token_ids = token_ids[: self.seq_len]
                yield torch.tensor(token_ids, dtype=torch.long)


def build_dataloader(tokenizer, seq_len: int, batch_size: int, cfg: SeKVConfig) -> DataLoader:
    """Build streaming dataloader of fixed-length token sequences."""
    dataset = RedPajamaLongDocuments(
        tokenizer=tokenizer,
        seq_len=seq_len,
        subsets=("arxiv", "book", "github"),
        seed=0,
        min_doc_tokens=seq_len,
    )

    def collate(batch: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(batch, dim=0)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )


def curriculum_schedule(max_steps: int, curriculum: list[int]) -> list[tuple[int, int]]:
    """Return (start_step, seq_len) stage boundaries split roughly equally over steps."""
    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0, got {max_steps}")
    if not curriculum:
        raise ValueError("curriculum must not be empty")

    n = len(curriculum)
    base = max_steps // n
    rem = max_steps % n

    schedule: list[tuple[int, int]] = []
    start = 0
    for i, seq_len in enumerate(curriculum):
        schedule.append((start, int(seq_len)))
        stage_steps = base + (1 if i < rem else 0)
        start += stage_steps
    return schedule
