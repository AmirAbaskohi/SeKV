from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Optional, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hf_id: Optional[str]
    dtype: str
    device_map: str
    attn_implementation: str


@dataclass(frozen=True)
class SegmentationConfig:
    alpha: float
    l_min: int


@dataclass(frozen=True)
class MemoryConfig:
    r_max: int
    summary_dim: int


@dataclass(frozen=True)
class RoutingConfig:
    tau_init: float
    ste_temperature: float


@dataclass(frozen=True)
class LossConfig:
    lambda_zoom: float
    lambda_recon: float
    lambda_budget: float
    beta: float
    w_pos: float
    rho: float


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    weight_decay: float
    warmup_ratio: float
    max_steps: int
    batch_size: int
    curriculum: list[int]
    mining_window: int
    budget_anneal_frac: float


@dataclass(frozen=True)
class PathsConfig:
    output_dir: str
    data_dir: str
    checkpoint_dir: str


@dataclass(frozen=True)
class SeKVConfig:
    model: ModelConfig
    segmentation: SegmentationConfig
    memory: MemoryConfig
    routing: RoutingConfig
    loss: LossConfig
    train: TrainConfig
    paths: PathsConfig


@dataclass(frozen=True)
class EvalConfig:
    benchmarks: list[str]
    data_dir: str
    max_samples_per_task: Optional[int]
    budgets: list[Optional[int]]
    tensor_parallel_threshold: int
    output_dir: str
    longbench_dataset: str
    longbench_tasks: list[str]
    infinitebench_dataset: str
    gsm8k_dataset: str
    ruler_lengths: list[int]
    niah_lengths: list[int]
    niah_depths: list[float]
    random_seed: int
    gsm8k_few_shot: bool
    gsm8k_few_shot_k: int


def _set_dotted_value(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not parts or any(part == "" for part in parts):
        raise KeyError(f"Invalid override key: {dotted_key}")

    cursor: Any = data
    for part in parts[:-1]:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise KeyError(f"Unknown override key: {dotted_key}")
        cursor = cursor[part]

    leaf = parts[-1]
    if not isinstance(cursor, Mapping) or leaf not in cursor:
        raise KeyError(f"Unknown override key: {dotted_key}")

    cursor[leaf] = value


def _coerce_scalar(value: Any, expected: type, path: str) -> Any:
    if expected is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path} must be bool, got {type(value).__name__}")
        return value

    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be int, got {type(value).__name__}")
        return value

    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{path} must be float, got {type(value).__name__}")
        return float(value)

    if expected is str:
        if not isinstance(value, str):
            raise TypeError(f"{path} must be str, got {type(value).__name__}")
        return value

    raise TypeError(f"Unsupported type annotation at {path}: {expected}")


def _coerce_value(value: Any, annotation: Any, path: str) -> Any:
    origin = get_origin(annotation)

    if annotation is Any:
        return value

    if is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping, got {type(value).__name__}")
        return _build_dataclass(annotation, value, path)

    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a list, got {type(value).__name__}")
        (item_type,) = get_args(annotation)
        return [_coerce_value(item, item_type, f"{path}[{idx}]") for idx, item in enumerate(value)]

    if origin is Union:
        args = get_args(annotation)
        if type(None) in args:
            if value is None:
                return None
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) != 1:
                raise TypeError(f"Unsupported optional annotation at {path}: {annotation}")
            return _coerce_value(value, non_none_args[0], path)

        errors: list[str] = []
        for arg in args:
            try:
                return _coerce_value(value, arg, path)
            except TypeError as err:
                errors.append(str(err))
        raise TypeError(f"{path} failed Union validation: {'; '.join(errors)}")

    if isinstance(annotation, type):
        return _coerce_scalar(value, annotation, path)

    raise TypeError(f"Unsupported annotation at {path}: {annotation}")


def _build_dataclass(dataclass_type: type[Any], data: Mapping[str, Any], path: str) -> Any:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(data).__name__}")

    field_defs = {field.name: field for field in fields(dataclass_type)}
    type_hints = get_type_hints(dataclass_type)
    unknown = [key for key in data.keys() if key not in field_defs]
    if unknown:
        unknown_sorted = ", ".join(sorted(unknown))
        raise KeyError(f"Unknown key(s) under {path}: {unknown_sorted}")

    kwargs: dict[str, Any] = {}
    for field_name, field_def in field_defs.items():
        field_path = f"{path}.{field_name}" if path else field_name
        if field_name not in data:
            raise KeyError(f"Missing required key: {field_path}")
        kwargs[field_name] = _coerce_value(data[field_name], type_hints[field_name], field_path)

    return dataclass_type(**kwargs)


def load_config(path: str, overrides: dict | None = None) -> SeKVConfig:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        loaded = {}

    if not isinstance(loaded, dict):
        raise TypeError(f"Top-level config must be a mapping, got {type(loaded).__name__}")

    data: dict[str, Any] = dict(loaded)

    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError(f"overrides must be a dict, got {type(overrides).__name__}")
        for dotted_key, value in overrides.items():
            if not isinstance(dotted_key, str):
                raise TypeError("All override keys must be strings")
            _set_dotted_value(data, dotted_key, value)

    return _build_dataclass(SeKVConfig, data, "config")


def load_eval_config(path: str, overrides: dict | None = None) -> EvalConfig:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Top-level eval config must be a mapping, got {type(loaded).__name__}")

    if "eval" in loaded:
        eval_data = loaded["eval"]
        if not isinstance(eval_data, dict):
            raise TypeError(f"'eval' section must be a mapping, got {type(eval_data).__name__}")
        data: dict[str, Any] = dict(eval_data)
    else:
        data = dict(loaded)

    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError(f"overrides must be a dict, got {type(overrides).__name__}")
        for dotted_key, value in overrides.items():
            if not isinstance(dotted_key, str):
                raise TypeError("All override keys must be strings")
            _set_dotted_value(data, dotted_key, value)

    return _build_dataclass(EvalConfig, data, "eval")
