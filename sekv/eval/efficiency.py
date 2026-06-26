from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class EfficiencyRecord:
    peak_resident_kv_tokens: int
    mean_resident_kv_tokens: float
    mean_expansion_rate: float
    mean_effective_rank: float
    budget: int | None
    seq_len: int
    peak_gpu_mem_bytes: int


class ResidencyAccumulator:
    def __init__(self):
        self.resident_tokens: list[int] = []
        self.expansion_rates: list[float] = []
        self.effective_ranks: list[float] = []

    def add(self, resident_kv_tokens: int, expansion_rate: float, effective_rank: float) -> None:
        self.resident_tokens.append(int(resident_kv_tokens))
        self.expansion_rates.append(float(expansion_rate))
        self.effective_ranks.append(float(effective_rank))

    def summary(self, budget: int | None, seq_len: int, peak_gpu_mem_bytes: int) -> EfficiencyRecord:
        peak = max(self.resident_tokens) if self.resident_tokens else 0
        mean_resident = (
            float(sum(self.resident_tokens)) / float(len(self.resident_tokens))
            if self.resident_tokens
            else 0.0
        )
        mean_expand = (
            float(sum(self.expansion_rates)) / float(len(self.expansion_rates))
            if self.expansion_rates
            else 0.0
        )
        mean_rank = (
            float(sum(self.effective_ranks)) / float(len(self.effective_ranks))
            if self.effective_ranks
            else 0.0
        )

        if budget is not None and peak > budget:
            raise AssertionError(
                f"Measured peak resident KV tokens ({peak}) exceeds budget ({budget})"
            )

        return EfficiencyRecord(
            peak_resident_kv_tokens=peak,
            mean_resident_kv_tokens=mean_resident,
            mean_expansion_rate=mean_expand,
            mean_effective_rank=mean_rank,
            budget=budget,
            seq_len=seq_len,
            peak_gpu_mem_bytes=int(peak_gpu_mem_bytes),
        )


def make_residency_hook() -> tuple[Callable[[int, float, float], None], ResidencyAccumulator]:
    acc = ResidencyAccumulator()

    def callback(resident_kv_tokens: int, expansion_rate: float, effective_rank: float) -> None:
        acc.add(resident_kv_tokens, expansion_rate, effective_rank)

    return callback, acc


@dataclass
class EfficiencyHandle:
    callback: Callable[[int, float, float], None]
    accumulator: ResidencyAccumulator
    record: EfficiencyRecord | None = None


@contextmanager
def track_efficiency(budget: int | None, seq_len: int):
    callback, acc = make_residency_hook()
    handle = EfficiencyHandle(callback=callback, accumulator=acc)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        base_alloc = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        base_alloc = 0

    try:
        yield handle
    finally:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            peak_abs = int(torch.cuda.max_memory_allocated(device))
            peak_delta = max(0, peak_abs - base_alloc)
        else:
            peak_delta = 0

        handle.record = acc.summary(
            budget=budget,
            seq_len=int(seq_len),
            peak_gpu_mem_bytes=peak_delta,
        )
