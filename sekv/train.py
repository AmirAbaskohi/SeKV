from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from sekv.arch import ArchAdapter, build_arch_adapter
from sekv.attention import decode_step_layer
from sekv.backbone import FrozenBackbone
from sekv.config import SeKVConfig, TrainConfig, load_config
from sekv.data import RedPajamaLongDocuments, curriculum_schedule
from sekv.losses import budget_loss, distill_loss, recon_loss, total_loss, zoom_loss
from sekv.memory import SpanMemory, build_span_memory, derive_layer, memory_view, reconstruct_span
from sekv.modules import KIND_KEY, KIND_VALUE, SeKVModules
from sekv.rope import apply_rope
from sekv.segmentation import memory_layout, segment_sequence
from sekv.teacher import build_teacher_signals, candidate_query_positions, coverage_targets


def build_optimizer(modules: SeKVModules, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW over trainable lightweight SeKV modules only."""
    return AdamW(modules.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainConfig):
    """Cosine decay with linear warmup."""
    warmup_steps = int(cfg.max_steps * cfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, cfg.max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def budget_anneal(step: int, max_steps: int, cfg: SeKVConfig) -> float:
    """Linear ramp of lambda_budget from 0 over the first budget_anneal_frac of training, then hold."""
    del max_steps
    ramp_steps = max(1, int(cfg.train.max_steps * float(cfg.train.budget_anneal_frac)))
    if step >= ramp_steps:
        return float(cfg.loss.lambda_budget)
    return float(cfg.loss.lambda_budget) * (float(step) / float(ramp_steps))


def _autocast(backbone: FrozenBackbone):
    device = backbone.model.device
    if backbone.model.dtype == torch.bfloat16:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


@dataclass
class _TrainLayerState:
    """Per-sequence, query-independent tensors of one layer (shared by all sampled positions)."""

    comp_ids: list[int]
    summaries: torch.Tensor
    fine_k: torch.Tensor
    fine_v: torch.Tensor
    token_span: torch.Tensor
    fine_cum: list[int]
    gates_k: torch.Tensor
    gates_v: torch.Tensor
    sv_k: torch.Tensor
    sv_v: torch.Tensor
    energy_k: torch.Tensor
    energy_v: torch.Tensor


def _prepare_layer_states(
    memory: SpanMemory,
    modules: SeKVModules,
    adapter: ArchAdapter,
    device: torch.device,
) -> list[_TrainLayerState]:
    """Routing summaries, learned rank gates, and gated reconstructions of every compressed span."""
    spec = memory.spec
    r_max = modules.r_max
    states: list[_TrainLayerState] = []

    for layer_idx, layer_mem in enumerate(memory.layers):
        derived = derive_layer(layer_mem)
        comp = [layer_mem.spans[i] for i in derived.comp_ids]
        m = len(comp)
        k_bar = derived.k_bar.to(device)
        summaries = modules.routing_summaries(layer_idx, k_bar) if m else torch.zeros((0, spec.num_q_heads, modules.summary_dim), device=device)

        if m:
            sv_k = torch.stack([F.pad(s.factors.s_k, (0, r_max - s.factors.s_k.size(-1))) for s in comp]).to(device)
            sv_v = torch.stack([F.pad(s.factors.s_v, (0, r_max - s.factors.s_v.size(-1))) for s in comp]).to(device)
            valid = torch.tensor([s.factors.s_k.size(-1) for s in comp], dtype=torch.long, device=device).view(-1, 1)
            gates_k = modules.rank_gates(sv_k, k_bar, valid, KIND_KEY)
            gates_v = modules.rank_gates(sv_v, k_bar, valid, KIND_VALUE)
            energy_k = torch.stack([s.factors.energy_k for s in comp]).to(device)
            energy_v = torch.stack([s.factors.energy_v for s in comp]).to(device)

            positions = torch.cat([torch.arange(s.start, s.stop, device=device) for s in comp])
            cos, sin = adapter.rope_cos_sin(positions)
            k_parts, v_parts, offset = [], [], 0
            for i, span in enumerate(comp):
                f = span.factors
                k_pre = reconstruct_span(f.U_k.to(device), f.s_k.to(device), f.Vmat_k.to(device), gates_k[i])
                k_parts.append(apply_rope(k_pre, cos[offset : offset + span.length], sin[offset : offset + span.length]))
                v_parts.append(reconstruct_span(f.U_v.to(device), f.s_v.to(device), f.Vmat_v.to(device), gates_v[i]))
                offset += span.length
            fine_k = torch.cat(k_parts, dim=1)
            fine_v = torch.cat(v_parts, dim=1)
            lengths = [s.length for s in comp]
            token_span = torch.repeat_interleave(torch.arange(m, device=device), torch.tensor(lengths, device=device))
            fine_cum = torch.cumsum(torch.tensor(lengths), dim=0).tolist()
        else:
            empty = torch.zeros((0, spec.num_kv_heads, r_max), device=device)
            sv_k = sv_v = gates_k = gates_v = empty
            energy_k = energy_v = torch.zeros((0, spec.num_kv_heads), device=device)
            fine_k = fine_v = torch.zeros((spec.num_kv_heads, 0, spec.head_dim), device=device)
            token_span = torch.zeros((0,), dtype=torch.long, device=device)
            fine_cum = []

        states.append(
            _TrainLayerState(
                comp_ids=list(derived.comp_ids),
                summaries=summaries,
                fine_k=fine_k,
                fine_v=fine_v,
                token_span=token_span,
                fine_cum=fine_cum,
                gates_k=gates_k,
                gates_v=gates_v,
                sv_k=sv_k,
                sv_v=sv_v,
                energy_k=energy_k,
                energy_v=energy_v,
            )
        )
    return states


def _student_forward(
    backbone: FrozenBackbone,
    adapter: ArchAdapter,
    modules: SeKVModules,
    view: SpanMemory,
    states: list[_TrainLayerState],
    position: int,
    token_id: torch.Tensor,
    span_mass: torch.Tensor,
    cfg: SeKVConfig,
    ste_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SeKV decode step for the token at `position`, attending to its causal memory view."""
    spec = backbone.spec
    x = adapter.embed_tokens(token_id.view(1)).reshape(-1)

    alphas: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    zs: list[torch.Tensor] = []
    fine: list[torch.Tensor] = []

    for layer_idx in range(spec.num_layers):
        layer_mem = view.layers[layer_idx]
        state = states[layer_idx]

        attn_in = adapter.input_layernorm(layer_idx, x)
        q, k, v = adapter.project_qkv(layer_idx, attn_in)
        q = adapter.apply_rope(q, position)
        k = adapter.apply_rope(k, position)
        layer_mem.append_window(k, v)

        m = len(derive_layer(layer_mem).comp_ids)
        n_tok = state.fine_cum[m - 1] if m else 0
        out, info = decode_step_layer(
            q_layer=q,
            layer_mem=layer_mem,
            modules=modules,
            layer=layer_idx,
            spec=spec,
            cfg=cfg,
            budget=None,
            training=True,
            ste_temperature=ste_temperature,
            summaries=state.summaries[:m],
            train_fine=(state.fine_k[:, :n_tok], state.fine_v[:, :n_tok], state.token_span[:n_tok]),
            return_info=True,
        )
        x = adapter.decoder_layer_forward(layer_idx, x, adapter.project_attention_output(layer_idx, out))

        if m:
            comp_idx = torch.tensor(state.comp_ids[:m], dtype=torch.long)
            y = coverage_targets(span_mass[layer_idx].index_select(-1, comp_idx), rho=cfg.loss.rho)
            alphas.append(info.alpha.reshape(-1))
            targets.append(y.transpose(0, 1).reshape(-1).to(info.alpha.device))
            zs.append(info.z.reshape(-1))
            fine.append(info.head_fine.reshape(-1).to(torch.float32))

    logits = adapter.lm_head(adapter.final_norm(x)).reshape(-1)

    def _cat(parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(parts) if parts else torch.zeros((0,), device=logits.device)

    return logits, {"alpha": _cat(alphas), "y": _cat(targets), "z": _cat(zs), "fine": _cat(fine)}


def _sequence_loss(
    backbone: FrozenBackbone,
    adapter: ArchAdapter,
    modules: SeKVModules,
    seq: torch.Tensor,
    cfg: SeKVConfig,
    budget_weight: float,
    ste_temperature: float,
) -> Optional[tuple[torch.Tensor, dict[str, float]]]:
    spec = backbone.spec
    device = backbone.model.device
    window = int(cfg.memory.local_window)

    candidates = candidate_query_positions(int(seq.numel()), cfg)
    if not candidates:
        return None

    with torch.no_grad(), _autocast(backbone):
        prefill = backbone.prefill(seq.unsqueeze(0), keep_hidden=True, query_positions=candidates)
    seg = segment_sequence(prefill.surprisal, cfg.segmentation)
    teacher = build_teacher_signals(backbone, prefill, seg, cfg, adapter.rope_cos_sin)
    if not teacher.query_positions:
        return None

    layout_max = memory_layout(seg, max(teacher.query_positions), window)
    if not any(not s.full_resolution for s in seg.spans[: layout_max.num_spans]):
        return None
    with torch.no_grad():
        memory = build_span_memory(prefill, seg, layout_max, cfg, spec, adapter.rope_cos_sin, modules=None, offload=False)

    with _autocast(backbone):
        states = _prepare_layer_states(memory, modules, adapter, device)

        student_rows, teacher_rows = [], []
        alpha_rows, y_rows, z_rows, fine_rows = [], [], [], []
        for pos in teacher.query_positions:
            layout = memory_layout(seg, pos, window)
            view = memory_view(memory, layout, prefill, window_stop=pos)
            logits, info = _student_forward(
                backbone, adapter, modules, view, states, pos, seq[pos].to(device),
                teacher.span_mass[pos], cfg, ste_temperature,
            )
            student_rows.append(logits)
            teacher_rows.append(teacher.teacher_logits[pos].to(device))
            alpha_rows.append(info["alpha"])
            y_rows.append(info["y"])
            z_rows.append(info["z"])
            fine_rows.append(info["fine"])

    def _flat(name: str) -> torch.Tensor:
        return torch.cat([getattr(s, name).reshape(-1, *getattr(s, name).shape[2:]) for s in states], dim=0)

    gates_k, gates_v = _flat("gates_k"), _flat("gates_v")
    parts = {
        "distill": distill_loss(torch.stack(student_rows), torch.stack(teacher_rows)),
        "zoom": zoom_loss(torch.cat(alpha_rows), torch.cat(y_rows), w_pos=cfg.loss.w_pos),
        "recon": recon_loss(gates_k, _flat("sv_k"), gates_v, _flat("sv_v"), _flat("energy_k").reshape(-1), _flat("energy_v").reshape(-1)),
        "budget": budget_loss(torch.cat(z_rows), gates_k, gates_v, beta=cfg.loss.beta),
    }
    loss, logs = total_loss(parts=parts, weights=cfg.loss, budget_weight=budget_weight)

    with torch.no_grad():
        logs["mean_effective_rank"] = float(((gates_k > cfg.memory.gate_eps).sum(-1).float().mean() + (gates_v > cfg.memory.gate_eps).sum(-1).float().mean()).item() / 2.0)
        fine_all = torch.cat(fine_rows)
        logs["mean_expansion_rate"] = float(fine_all.mean().item()) if fine_all.numel() else 0.0
    return loss, logs


def train_step(
    backbone: FrozenBackbone,
    adapter: ArchAdapter,
    modules: SeKVModules,
    input_ids: torch.Tensor,
    cfg: SeKVConfig,
    step: int,
    ste_temperature: float,
) -> dict[str, float]:
    """One optimization step over the lightweight SeKV modules (gradients accumulated per sequence)."""
    if not hasattr(modules, "_optimizer") or not hasattr(modules, "_scheduler"):
        raise RuntimeError("train_step expects optimizer and scheduler attached on modules")

    optimizer = modules._optimizer
    scheduler = modules._scheduler

    modules.train()
    optimizer.zero_grad(set_to_none=True)

    if input_ids.ndim == 1:
        batch_ids = input_ids.unsqueeze(0)
    elif input_ids.ndim == 2:
        batch_ids = input_ids
    else:
        raise ValueError(f"input_ids must be [seq_len] or [batch,seq_len], got {tuple(input_ids.shape)}")
    batch_ids = batch_ids.to(device=backbone.model.device, dtype=torch.long)

    budget_weight = budget_anneal(step=step, max_steps=cfg.train.max_steps, cfg=cfg)
    results = []
    for seq in batch_ids:
        result = _sequence_loss(backbone, adapter, modules, seq, cfg, budget_weight, ste_temperature)
        if result is None:
            continue
        loss, logs = result
        (loss / float(batch_ids.size(0))).backward()
        results.append(logs)
        del loss

    keys = ["loss", "loss_distill", "loss_zoom", "loss_recon", "loss_budget", "mean_effective_rank", "mean_expansion_rate"]
    if not results:
        return {key: 0.0 for key in keys}

    torch.nn.utils.clip_grad_norm_(modules.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    out = {key: sum(r[key] for r in results) / len(results) for key in keys}
    out["budget_weight"] = budget_weight
    return out


def checkpoint_dir(cfg: SeKVConfig) -> Path:
    return Path(cfg.paths.checkpoint_dir) / cfg.model.name


def save_checkpoint(modules: SeKVModules, backbone: FrozenBackbone, cfg: SeKVConfig, step: int, name: str) -> Path:
    """Save trainable parameters with the config and model spec as plain dicts."""
    ckpt_dir = checkpoint_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": int(step),
        "model_spec": asdict(backbone.spec),
        "config": asdict(cfg),
        "trainable_parameters": {
            name: tensor.detach().cpu() for name, tensor in modules.trainable_parameters().items()
        },
    }
    path = ckpt_dir / name
    torch.save(payload, path)
    return path


def train(cfg: SeKVConfig) -> None:
    """Train SeKV lightweight modules with streaming data and a length curriculum."""
    backbone = FrozenBackbone(cfg.model)
    adapter = build_arch_adapter(backbone)
    modules = SeKVModules(spec=backbone.spec, routing=cfg.routing, memory=cfg.memory)
    modules.to(device=backbone.model.device)

    optimizer = build_optimizer(modules, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train)
    modules._optimizer = optimizer
    modules._scheduler = scheduler

    schedule = curriculum_schedule(cfg.train.max_steps, cfg.train.curriculum)
    stage_ptr = 0
    seq_len = schedule[0][1]

    dataset = RedPajamaLongDocuments(
        tokenizer=backbone.tokenizer,
        seq_len=seq_len,
        subsets=("arxiv", "book", "github"),
        seed=0,
        min_doc_tokens=seq_len,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        collate_fn=lambda batch: torch.stack(batch, dim=0),
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    data_iter = iter(dataloader)

    log_interval = 10
    ckpt_interval = 200

    for step in range(cfg.train.max_steps):
        while stage_ptr + 1 < len(schedule) and step >= schedule[stage_ptr + 1][0]:
            stage_ptr += 1
            seq_len = schedule[stage_ptr][1]
            dataset.set_seq_len(seq_len)
            dataset.min_doc_tokens = seq_len
            data_iter = iter(dataloader)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        logs = train_step(
            backbone=backbone,
            adapter=adapter,
            modules=modules,
            input_ids=batch,
            cfg=cfg,
            step=step,
            ste_temperature=cfg.routing.ste_temperature,
        )

        if step % log_interval == 0:
            print(
                f"step={step:06d} seq={seq_len} "
                f"loss={logs.get('loss', 0.0):.4f} "
                f"distill={logs.get('loss_distill', 0.0):.4f} "
                f"zoom={logs.get('loss_zoom', 0.0):.4f} "
                f"recon={logs.get('loss_recon', 0.0):.4f} "
                f"budget={logs.get('loss_budget', 0.0):.4f} "
                f"rank={logs.get('mean_effective_rank', 0.0):.3f} "
                f"expand={logs.get('mean_expansion_rate', 0.0):.3f}"
            )

        if (step + 1) % ckpt_interval == 0:
            save_checkpoint(modules, backbone, cfg, step + 1, f"step_{step + 1:07d}.pt")

    path = save_checkpoint(modules, backbone, cfg, cfg.train.max_steps, "final.pt")
    print(f"Saved final checkpoint to {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SeKV lightweight modules")
    parser.add_argument("--config", type=str, default="sekv/configs/default.yaml")
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    if args.model is not None:
        overrides["model.name"] = args.model
    return overrides or None


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, overrides=_build_overrides(args))

    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)

    train(cfg)


if __name__ == "__main__":
    main()
