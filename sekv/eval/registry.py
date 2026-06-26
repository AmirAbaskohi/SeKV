from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from sekv.config import EvalConfig


@dataclass
class BenchmarkSample:
    sample_id: str
    prompt: str
    reference: object
    max_new_tokens: int
    meta: dict


class Benchmark(Protocol):
    name: str

    def load(self, cfg: EvalConfig) -> Iterable[BenchmarkSample]:
        ...

    def score(self, sample: BenchmarkSample, prediction: str) -> dict:
        ...

    def aggregate(self, per_sample: list[dict]) -> dict:
        ...


BENCHMARK_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls: type):
        if name in BENCHMARK_REGISTRY:
            raise KeyError(f"Benchmark '{name}' already registered")
        BENCHMARK_REGISTRY[name] = cls
        return cls

    return decorator


def get_benchmark(name: str) -> Benchmark:
    if name not in BENCHMARK_REGISTRY:
        known = ", ".join(sorted(BENCHMARK_REGISTRY.keys()))
        raise KeyError(f"Unknown benchmark '{name}'. Known: {known}")
    cls = BENCHMARK_REGISTRY[name]
    obj = cls()
    return obj
