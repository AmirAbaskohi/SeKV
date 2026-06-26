from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sekv.backbone import FrozenBackbone
from sekv.config import EvalConfig, ModelConfig, SeKVConfig, load_config, load_eval_config
from sekv.eval.efficiency import EfficiencyRecord, track_efficiency
from sekv.eval.registry import Benchmark, get_benchmark
from sekv.generate import GenerationConfig, generate_text, load_sekv_checkpoint

# Import side-effect registrations.
from sekv.eval import benchmarks as _benchmarks  # noqa: F401


def select_device_strategy(seq_len: int, cfg: EvalConfig) -> str:
    """Single-GPU by default; tensor-parallel strategy for very long sequences."""
    if seq_len > cfg.tensor_parallel_threshold:
        return "tensor_parallel"
    return "single"


def build_backbone_for_eval(cfg: SeKVConfig, eval_cfg: EvalConfig, seq_len: int) -> FrozenBackbone:
    """Build backbone for single or TP-like placement policy."""
    strategy = select_device_strategy(seq_len=seq_len, cfg=eval_cfg)

    if strategy == "single":
        if torch.cuda.is_available():
            device_map = "cuda:0"
        else:
            device_map = "cpu"
    else:
        # TP-like fallback via automatic sharding placement.
        device_map = "auto"

    model_cfg = ModelConfig(
        name=cfg.model.name,
        hf_id=cfg.model.hf_id,
        dtype=cfg.model.dtype,
        device_map=device_map,
        attn_implementation=cfg.model.attn_implementation,
    )
    return FrozenBackbone(model_cfg)


def _estimate_prompt_len(backbone: FrozenBackbone, prompt: str, apply_chat_template: bool) -> int:
    tok = backbone.tokenizer
    if backbone.spec.is_instruct and apply_chat_template and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    else:
        ids = tok(prompt, return_tensors="pt")["input_ids"]
    return int(ids.size(-1))


def _summarize_efficiency(records: list[EfficiencyRecord], budget: int | None) -> dict:
    if not records:
        return {
            "peak_resident_kv_tokens": 0,
            "mean_resident_kv_tokens": 0.0,
            "mean_expansion_rate": 0.0,
            "mean_effective_rank": 0.0,
            "peak_gpu_mem_bytes": 0,
            "budget": budget,
        }

    peak = max(r.peak_resident_kv_tokens for r in records)
    mean_resident = sum(r.mean_resident_kv_tokens for r in records) / len(records)
    mean_expand = sum(r.mean_expansion_rate for r in records) / len(records)
    mean_rank = sum(r.mean_effective_rank for r in records) / len(records)
    peak_mem = max(r.peak_gpu_mem_bytes for r in records)

    if budget is not None and peak > budget:
        raise AssertionError(f"Peak residency {peak} exceeded budget {budget}")

    return {
        "peak_resident_kv_tokens": int(peak),
        "mean_resident_kv_tokens": float(mean_resident),
        "mean_expansion_rate": float(mean_expand),
        "mean_effective_rank": float(mean_rank),
        "peak_gpu_mem_bytes": int(peak_mem),
        "budget": budget,
    }


def evaluate_benchmark(
    backbone,
    modules,
    benchmark: Benchmark,
    cfg: SeKVConfig,
    eval_cfg: EvalConfig,
    budget: int | None,
) -> dict:
    """Run one benchmark under one budget and return metrics + per-sample efficiency."""
    per_sample_rows: list[dict] = []
    eff_records: list[EfficiencyRecord] = []

    apply_chat_template = backbone.spec.is_instruct

    for sample in benchmark.load(eval_cfg):
        seq_len = _estimate_prompt_len(backbone, sample.prompt, apply_chat_template)
        strategy = select_device_strategy(seq_len=seq_len, cfg=eval_cfg)
        gen_cfg = GenerationConfig(
            max_new_tokens=int(sample.max_new_tokens),
            budget=budget,
            eos_token_id=backbone.tokenizer.eos_token_id,
        )

        track_budget = budget if strategy == "single" else None
        with track_efficiency(budget=track_budget, seq_len=seq_len) as handle:
            prediction = generate_text(
                backbone=backbone,
                modules=modules,
                prompt=sample.prompt,
                cfg=cfg,
                gen_cfg=gen_cfg,
                apply_chat_template=apply_chat_template,
                residency_callback=handle.callback if strategy == "single" else None,
            )

        if handle.record is None:
            raise RuntimeError("Efficiency tracking did not produce a record")

        row = benchmark.score(sample, prediction)
        row["sample_id"] = sample.sample_id
        row["prompt_meta"] = sample.meta
        row["device_strategy"] = strategy
        row["prediction"] = prediction
        row["efficiency"] = asdict(handle.record)

        per_sample_rows.append(row)
        if strategy == "single":
            eff_records.append(handle.record)

    return {
        "aggregate": benchmark.aggregate(per_sample_rows),
        "efficiency": _summarize_efficiency(eff_records, budget=budget),
        "per_sample": per_sample_rows,
    }


def run_eval(cfg: SeKVConfig, eval_cfg: EvalConfig, checkpoint: str) -> dict:
    """Run all requested benchmark x budget combinations and persist JSON outputs."""
    out_dir = Path(eval_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}

    for bench_name in eval_cfg.benchmarks:
        benchmark = get_benchmark(bench_name)

        seq_len_probe = eval_cfg.tensor_parallel_threshold + 1 if bench_name == "infinitebench" else 0
        backbone = build_backbone_for_eval(cfg=cfg, eval_cfg=eval_cfg, seq_len=seq_len_probe)
        modules = load_sekv_checkpoint(path=checkpoint, backbone=backbone)

        bench_results: dict[str, Any] = {}
        for budget in eval_cfg.budgets:
            key = "none" if budget is None else str(int(budget))
            result = evaluate_benchmark(
                backbone=backbone,
                modules=modules,
                benchmark=benchmark,
                cfg=cfg,
                eval_cfg=eval_cfg,
                budget=budget,
            )
            bench_results[key] = result

            out_path = out_dir / f"{bench_name}_budget_{key}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        all_results[bench_name] = bench_results

    combined_path = out_dir / "combined_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results


def _parse_optional_budgets(raw: str | None) -> list[int | None] | None:
    if raw is None or raw.strip() == "":
        return None

    values: list[int | None] = []
    for part in raw.split(","):
        p = part.strip().lower()
        if p in {"none", "null"}:
            values.append(None)
        else:
            values.append(int(p))
    return values


def _parse_optional_benchmarks(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _compact_summary(results: dict) -> str:
    lines: list[str] = []
    for bench, budgets in results.items():
        for budget, payload in budgets.items():
            agg = payload.get("aggregate", {})
            eff = payload.get("efficiency", {})
            main_score = agg.get("macro_mean", agg.get("overall", agg.get("accuracy", None)))
            lines.append(
                f"{bench:14s} budget={budget:>5s} score={main_score} peak_kv={eff.get('peak_resident_kv_tokens', 0)}"
            )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SeKV benchmark evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmarks", type=str, default=None)
    parser.add_argument("--budgets", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


def _build_cfg_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    if args.model is not None:
        overrides["model.name"] = args.model
    return overrides or None


def _build_eval_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    benches = _parse_optional_benchmarks(args.benchmarks)
    budgets = _parse_optional_budgets(args.budgets)
    if benches is not None:
        overrides["benchmarks"] = benches
    if budgets is not None:
        overrides["budgets"] = budgets
    return overrides or None


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, overrides=_build_cfg_overrides(args))
    eval_cfg = load_eval_config(args.eval_config, overrides=_build_eval_overrides(args))

    os.makedirs(eval_cfg.output_dir, exist_ok=True)
    results = run_eval(cfg=cfg, eval_cfg=eval_cfg, checkpoint=args.checkpoint)
    print(_compact_summary(results))


if __name__ == "__main__":
    main()
