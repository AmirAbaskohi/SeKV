from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class EfficiencyRecord:
    # GPU-resident KV in tokens per KV head (peak over layers and steps); enforced by generate().
    peak_resident_kv_tokens: float
    mean_resident_kv_tokens: float
    mean_expansion_rate: float
    mean_effective_rank: float
    mean_h2d_bytes_per_step: float
    mean_reuse_rate: float
    budget: float | None
    seq_len: int
    peak_gpu_mem_bytes: int


def _mean(values: list[float]) -> float:
    return float(sum(values)) / float(len(values)) if values else 0.0


class ResidencyAccumulator:
    def __init__(self):
        self.resident_tokens: list[float] = []
        self.expansion_rates: list[float] = []
        self.effective_ranks: list[float] = []
        self.h2d_bytes: list[float] = []
        self.reuse_rates: list[float] = []

    def add(
        self,
        resident_kv_tokens: float,
        expansion_rate: float,
        effective_rank: float,
        h2d_bytes: int = 0,
        reuse_rate: float = 0.0,
    ) -> None:
        self.resident_tokens.append(float(resident_kv_tokens))
        self.expansion_rates.append(float(expansion_rate))
        self.effective_ranks.append(float(effective_rank))
        self.h2d_bytes.append(float(h2d_bytes))
        self.reuse_rates.append(float(reuse_rate))

    def summary(self, budget: float | None, seq_len: int, peak_gpu_mem_bytes: int) -> EfficiencyRecord:
        return EfficiencyRecord(
            peak_resident_kv_tokens=max(self.resident_tokens) if self.resident_tokens else 0.0,
            mean_resident_kv_tokens=_mean(self.resident_tokens),
            mean_expansion_rate=_mean(self.expansion_rates),
            mean_effective_rank=_mean(self.effective_ranks),
            mean_h2d_bytes_per_step=_mean(self.h2d_bytes),
            mean_reuse_rate=_mean(self.reuse_rates),
            budget=budget,
            seq_len=seq_len,
            peak_gpu_mem_bytes=int(peak_gpu_mem_bytes),
        )


def make_residency_hook() -> tuple[Callable[..., None], ResidencyAccumulator]:
    acc = ResidencyAccumulator()

    def callback(
        resident_kv_tokens: float,
        expansion_rate: float,
        effective_rank: float,
        h2d_bytes: int = 0,
        reuse_rate: float = 0.0,
    ) -> None:
        acc.add(resident_kv_tokens, expansion_rate, effective_rank, h2d_bytes, reuse_rate)

    return callback, acc


@dataclass
class EfficiencyHandle:
    callback: Callable[..., None]
    accumulator: ResidencyAccumulator
    record: EfficiencyRecord | None = None


@contextmanager
def track_efficiency(budget: float | None, seq_len: int):
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
            peak_delta = max(0, int(torch.cuda.max_memory_allocated(device)) - base_alloc)
        else:
            peak_delta = 0

        handle.record = acc.summary(budget=budget, seq_len=int(seq_len), peak_gpu_mem_bytes=peak_delta)
