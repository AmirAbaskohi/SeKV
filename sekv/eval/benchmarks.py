from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from sekv.config import EvalConfig
from sekv.eval.metrics import (
    LONGBENCH_TASK_METRIC,
    exact_match,
    gsm8k_answer_match,
    score_with,
    substring_match,
)
from sekv.eval.registry import Benchmark, BenchmarkSample, register


def _limit_for_task(items: list[BenchmarkSample], limit: int | None) -> list[BenchmarkSample]:
    if limit is None:
        return items
    return items[: max(0, int(limit))]


@register("longbench")
class LongBenchBenchmark(Benchmark):
    name = "longbench"

    def load(self, cfg: EvalConfig) -> Iterable[BenchmarkSample]:
        from datasets import load_dataset

        all_samples: list[BenchmarkSample] = []
        for task in cfg.longbench_tasks:
            ds = load_dataset(cfg.longbench_dataset, task, split="test")
            task_samples: list[BenchmarkSample] = []
            for i, item in enumerate(ds):
                context = item.get("context") or item.get("article") or item.get("input") or ""
                question = item.get("question") or item.get("input") or ""
                prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

                answers = item.get("answers")
                if isinstance(answers, list):
                    refs = [str(x) for x in answers]
                elif answers is None:
                    refs = [str(item.get("answer", ""))]
                else:
                    refs = [str(answers)]

                task_samples.append(
                    BenchmarkSample(
                        sample_id=f"{task}-{i}",
                        prompt=prompt,
                        reference=refs,
                        max_new_tokens=128,
                        meta={"task": task, "length_bucket": item.get("length", None)},
                    )
                )

            all_samples.extend(_limit_for_task(task_samples, cfg.max_samples_per_task))

        return all_samples

    def score(self, sample: BenchmarkSample, prediction: str) -> dict:
        task = str(sample.meta.get("task", ""))
        metric_key = LONGBENCH_TASK_METRIC.get(task, "qa_f1")
        refs = [str(x) for x in sample.reference]
        return {
            "sample_id": sample.sample_id,
            "task": task,
            "metric": metric_key,
            "score": float(score_with(metric_key, prediction, refs)),
        }

    def aggregate(self, per_sample: list[dict]) -> dict:
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in per_sample:
            by_task[str(row["task"])].append(float(row["score"]))

        task_means = {
            task: (sum(scores) / len(scores) if scores else 0.0)
            for task, scores in by_task.items()
        }
        macro = sum(task_means.values()) / len(task_means) if task_means else 0.0
        return {"task_mean": task_means, "macro_mean": macro}


@register("ruler")
class RulerBenchmark(Benchmark):
    name = "ruler"

    def load(self, cfg: EvalConfig) -> Iterable[BenchmarkSample]:
        rng = random.Random(cfg.random_seed)
        per_task = cfg.max_samples_per_task or 64

        samples: list[BenchmarkSample] = []
        for length in cfg.ruler_lengths:
            for i in range(per_task):
                key = f"KEY_{length}_{i}"
                value = f"VALUE_{rng.randint(10000, 99999)}"

                filler_tokens = [f"tok{rng.randint(0, 999)}" for _ in range(max(16, length // 8))]
                insert_at = rng.randint(0, len(filler_tokens))
                filler_tokens.insert(insert_at, f"{key}:{value}")
                context = " ".join(filler_tokens)

                prompt = (
                    "You are given a long synthetic context.\n"
                    f"Context:\n{context}\n\n"
                    f"Question: What is the value for key '{key}'?\n"
                    "Answer with the value only."
                )

                samples.append(
                    BenchmarkSample(
                        sample_id=f"ruler-{length}-{i}",
                        prompt=prompt,
                        reference=[value],
                        max_new_tokens=16,
                        meta={"task": "ruler_retrieval", "length": length},
                    )
                )

        return samples

    def score(self, sample: BenchmarkSample, prediction: str) -> dict:
        refs = [str(x) for x in sample.reference]
        return {
            "sample_id": sample.sample_id,
            "task": str(sample.meta.get("task", "ruler_retrieval")),
            "length": int(sample.meta.get("length", 0)),
            "metric": "exact_match",
            "score": float(exact_match(prediction, refs)),
            "substring": float(substring_match(prediction, refs)),
        }

    def aggregate(self, per_sample: list[dict]) -> dict:
        by_length: dict[int, list[float]] = defaultdict(list)
        for row in per_sample:
            by_length[int(row["length"])].append(float(row["score"]))
        length_mean = {
            str(length): (sum(vals) / len(vals) if vals else 0.0)
            for length, vals in by_length.items()
        }
        overall = sum(length_mean.values()) / len(length_mean) if length_mean else 0.0
        return {"length_mean": length_mean, "overall": overall}


@register("niah")
class NIAHBenchmark(Benchmark):
    name = "niah"

    def load(self, cfg: EvalConfig) -> Iterable[BenchmarkSample]:
        rng = random.Random(cfg.random_seed + 7)
        per_cell = cfg.max_samples_per_task or 16

        samples: list[BenchmarkSample] = []
        for length in cfg.niah_lengths:
            base_len = max(64, length // 8)
            for depth in cfg.niah_depths:
                for i in range(per_cell):
                    needle = f"NEEDLE-{length}-{depth:.2f}-{i}-{rng.randint(1000,9999)}"
                    hay = [f"w{rng.randint(0, 9999)}" for _ in range(base_len)]

                    insert_idx = min(len(hay), max(0, int(depth * len(hay))))
                    needle_sentence = f"The hidden answer is {needle}."
                    hay.insert(insert_idx, needle_sentence)

                    context = " ".join(hay)
                    prompt = (
                        "Read the long context and find the hidden answer.\n"
                        f"Context:\n{context}\n\n"
                        "Question: What is the hidden answer string?"
                    )

                    samples.append(
                        BenchmarkSample(
                            sample_id=f"niah-{length}-{depth:.2f}-{i}",
                            prompt=prompt,
                            reference=[needle],
                            max_new_tokens=24,
                            meta={"task": "niah", "length": length, "depth": depth},
                        )
                    )

        return samples

    def score(self, sample: BenchmarkSample, prediction: str) -> dict:
        refs = [str(x) for x in sample.reference]
        return {
            "sample_id": sample.sample_id,
            "task": "niah",
            "length": int(sample.meta.get("length", 0)),
            "depth": float(sample.meta.get("depth", 0.0)),
            "metric": "substring_match",
            "score": float(substring_match(prediction, refs)),
        }

    def aggregate(self, per_sample: list[dict]) -> dict:
        grid: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in per_sample:
            l = str(int(row["length"]))
            d = f"{float(row['depth']):.2f}"
            grid[l][d].append(float(row["score"]))

        mean_grid: dict[str, dict[str, float]] = {}
        for l, depths in grid.items():
            mean_grid[l] = {}
            for d, vals in depths.items():
                mean_grid[l][d] = sum(vals) / len(vals) if vals else 0.0

        flat = [v for depths in mean_grid.values() for v in depths.values()]
        overall = sum(flat) / len(flat) if flat else 0.0
        return {"grid": mean_grid, "overall": overall}


@register("infinitebench")
class InfiniteBenchBenchmark(Benchmark):
    name = "infinitebench"

    def load(self, cfg: EvalConfig) -> Iterable[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset(cfg.infinitebench_dataset, split="test")
        samples: list[BenchmarkSample] = []
        for i, item in enumerate(ds):
            task = str(item.get("task", "infinitebench"))
            prompt = str(item.get("prompt") or item.get("input") or "")
            if not prompt:
                context = str(item.get("context", ""))
                question = str(item.get("question", ""))
                prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

            answers = item.get("answers")
            if isinstance(answers, list):
                refs = [str(x) for x in answers]
            elif answers is not None:
                refs = [str(answers)]
            else:
                refs = [str(item.get("answer", ""))]

            samples.append(
                BenchmarkSample(
                    sample_id=f"infinitebench-{i}",
                    prompt=prompt,
                    reference=refs,
                    max_new_tokens=128,
                    meta={"task": task},
                )
            )

        return _limit_for_task(samples, cfg.max_samples_per_task)

    def score(self, sample: BenchmarkSample, prediction: str) -> dict:
        task = str(sample.meta.get("task", "infinitebench"))
        refs = [str(x) for x in sample.reference]

        if "math" in task.lower() or "numeric" in task.lower():
            metric = "gsm8k"
        elif "qa" in task.lower() or "question" in task.lower():
            metric = "qa_f1"
        else:
            metric = "exact_match"

        return {
            "sample_id": sample.sample_id,
            "task": task,
            "metric": metric,
            "score": float(score_with(metric, prediction, refs)),
        }

    def aggregate(self, per_sample: list[dict]) -> dict:
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in per_sample:
            by_task[str(row["task"])].append(float(row["score"]))

        task_mean = {
            task: (sum(vals) / len(vals) if vals else 0.0)
            for task, vals in by_task.items()
        }
        macro = sum(task_mean.values()) / len(task_mean) if task_mean else 0.0
        return {"task_mean": task_mean, "macro_mean": macro}


@register("gsm8k")
class GSM8KBenchmark(Benchmark):
    name = "gsm8k"

    _few_shot_examples = [
        (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are there?",
            "There are originally 3 cars. 2 more arrive. 3 + 2 = 5. #### 5",
        ),
        (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many are left?",
            "Together they had 32 + 42 = 74. After eating 35, 74 - 35 = 39. #### 39",
        ),
        (
            "Jason had 20 lollipops. He gave Denny some and now has 12. How many did he give?",
            "He had 20 and now has 12, so he gave 20 - 12 = 8. #### 8",
        ),
        (
            "Shawn has five toys. For Christmas, he gets two toys each from mom and dad. How many now?",
            "He gets 2 + 2 = 4 more. Total is 5 + 4 = 9. #### 9",
        ),
    ]

    def load(self, cfg: EvalConfig) -> Iterable[BenchmarkSample]:
        from datasets import load_dataset

        ds = load_dataset(cfg.gsm8k_dataset, "main", split="test")
        samples: list[BenchmarkSample] = []
        for i, item in enumerate(ds):
            q = str(item["question"])
            a = str(item["answer"])

            if cfg.gsm8k_few_shot:
                k = max(0, min(cfg.gsm8k_few_shot_k, len(self._few_shot_examples)))
                shots = self._few_shot_examples[:k]
                shot_block = "\n\n".join(
                    f"Q: {sq}\nA: {sa}" for sq, sa in shots
                )
                prompt = (
                    f"Solve the following math word problems.\n\n{shot_block}\n\n"
                    f"Q: {q}\nA:"
                )
            else:
                prompt = f"Solve the following math word problem.\nQ: {q}\nA:"

            samples.append(
                BenchmarkSample(
                    sample_id=f"gsm8k-{i}",
                    prompt=prompt,
                    reference=a,
                    max_new_tokens=256,
                    meta={"task": "gsm8k"},
                )
            )

        return _limit_for_task(samples, cfg.max_samples_per_task)

    def score(self, sample: BenchmarkSample, prediction: str) -> dict:
        return {
            "sample_id": sample.sample_id,
            "task": "gsm8k",
            "metric": "gsm8k",
            "score": float(gsm8k_answer_match(prediction, str(sample.reference))),
        }

    def aggregate(self, per_sample: list[dict]) -> dict:
        vals = [float(row["score"]) for row in per_sample]
        return {"accuracy": (sum(vals) / len(vals) if vals else 0.0)}
