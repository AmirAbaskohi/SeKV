from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from sekv.arch import ArchAdapter, build_arch_adapter
from sekv.attention import decode_step_layer, ste_zoom_gate
from sekv.backbone import FrozenBackbone
from sekv.config import SeKVConfig, TrainConfig, load_config
from sekv.data import RedPajamaLongDocuments, curriculum_schedule
from sekv.losses import budget_loss, distill_loss, recon_loss, total_loss, zoom_loss
from sekv.memory import SpanMemory, build_span_memory, effective_rank
from sekv.modules import SeKVModules
from sekv.segmentation import segment_prefill
from sekv.teacher import build_teacher_signals


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
    """Linear ramp of lambda_budget then hold constant."""
    del max_steps
    frac = float(cfg.train.budget_anneal_frac)
    ramp_steps = max(1, int(cfg.train.max_steps * frac))
    if step >= ramp_steps:
        return float(cfg.loss.lambda_budget)
    return float(cfg.loss.lambda_budget) * (float(step) / float(ramp_steps))


def _student_logits_for_position(
    backbone: FrozenBackbone,
    adapter: ArchAdapter,
    modules: SeKVModules,
    span_memory: SpanMemory,
    position: int,
    seq_hidden0: torch.Tensor,
    cfg: SeKVConfig,
    ste_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    spec = backbone.spec

    x = seq_hidden0[position].to(dtype=torch.float32)

    alpha_layers: list[torch.Tensor] = []
    z_layers: list[torch.Tensor] = []

    gates_k_list: list[torch.Tensor] = []
    sv_k_list: list[torch.Tensor] = []
    gates_v_list: list[torch.Tensor] = []
    sv_v_list: list[torch.Tensor] = []

    for layer_idx in range(spec.num_layers):
        layer_mem = span_memory.layers[layer_idx]

        attn_in = adapter.input_layernorm(layer_idx, x)
        q_heads, _k_heads, _v_heads = adapter.project_qkv(layer_idx, attn_in)
        q_heads = adapter.apply_rope_q(layer_idx, q_heads.to(dtype=torch.float32), position=position)

        attn_heads = decode_step_layer(
            q_layer=q_heads,
            layer_mem=layer_mem,
            modules=modules,
            layer=layer_idx,
            spec=spec,
            cfg=cfg,
            budget=None,
            training=True,
            ste_temperature=ste_temperature,
        )

        attn_out = adapter.project_attention_output(layer_idx, attn_heads)
        x = adapter.decoder_layer_forward(layer_idx, x, attn_out)

        non_full = [i for i, sp in enumerate(layer_mem.spans) if not sp.full_resolution]
        if non_full:
            k_bar = torch.stack([layer_mem.spans[i].k_bar for i in non_full], dim=0).to(
                device=x.device, dtype=torch.float32
            )
            span_len = torch.tensor(
                [layer_mem.spans[i].length for i in non_full],
                device=x.device,
                dtype=torch.float32,
            )
            q_proj_routing = modules.project_query(layer_idx, q_heads)
            K_bar = modules.routing_summaries(layer_idx, k_bar)
            alpha_comp = modules.relevance_gate(q_proj_routing, K_bar, span_len)
            z_comp = ste_zoom_gate(alpha_comp, modules.tau[layer_idx].to(x.device), ste_temperature)

            alpha_full = torch.zeros(
                (spec.num_q_heads, len(layer_mem.spans)), device=x.device, dtype=torch.float32
            )
            z_full = torch.zeros_like(alpha_full)
            for c_idx, s_idx in enumerate(non_full):
                alpha_full[:, s_idx] = alpha_comp[c_idx]
                z_full[:, s_idx] = (alpha_comp[c_idx] > modules.tau[layer_idx].to(x.device)).to(torch.float32)

            alpha_layers.append(alpha_full)
            z_layers.append(z_full)

            r_max = int(cfg.memory.r_max)
            gates_k_layer = torch.zeros(
                (spec.num_kv_heads, len(layer_mem.spans), r_max), device=x.device, dtype=torch.float32
            )
            gates_v_layer = torch.zeros_like(gates_k_layer)
            sv_k_layer = torch.zeros_like(gates_k_layer)
            sv_v_layer = torch.zeros_like(gates_k_layer)

            for s_idx in non_full:
                factors = layer_mem.spans[s_idx].factors
                if factors is None:
                    continue
                rank = int(factors.valid_rank)
                valid_rank = torch.tensor([rank], dtype=torch.long, device=x.device)

                for kv in range(spec.num_kv_heads):
                    ctx = layer_mem.spans[s_idx].k_bar[kv].to(device=x.device, dtype=torch.float32)
                    ctx = ctx / ctx.norm(p=2).clamp_min(1.0e-8)
                    ctx = ctx.unsqueeze(0)

                    svk_pad = torch.zeros((1, r_max), device=x.device, dtype=torch.float32)
                    svv_pad = torch.zeros((1, r_max), device=x.device, dtype=torch.float32)

                    svk = factors.s_k[kv].to(device=x.device, dtype=torch.float32)
                    svv = factors.s_v[kv].to(device=x.device, dtype=torch.float32)
                    svk_pad[:, :rank] = svk
                    svv_pad[:, :rank] = svv

                    gk = modules.g_phi(svk_pad, ctx, valid_rank).squeeze(0)
                    gv = modules.g_phi(svv_pad, ctx, valid_rank).squeeze(0)

                    gates_k_layer[kv, s_idx] = gk
                    gates_v_layer[kv, s_idx] = gv
                    sv_k_layer[kv, s_idx] = svk_pad.squeeze(0)
                    sv_v_layer[kv, s_idx] = svv_pad.squeeze(0)

            gates_k_list.append(gates_k_layer)
            gates_v_list.append(gates_v_layer)
            sv_k_list.append(sv_k_layer)
            sv_v_list.append(sv_v_layer)
        else:
            alpha_layers.append(
                torch.zeros((spec.num_q_heads, len(layer_mem.spans)), device=x.device, dtype=torch.float32)
            )
            z_layers.append(
                torch.zeros((spec.num_q_heads, len(layer_mem.spans)), device=x.device, dtype=torch.float32)
            )
            gates_k_list.append(
                torch.zeros((spec.num_kv_heads, len(layer_mem.spans), cfg.memory.r_max), device=x.device)
            )
            gates_v_list.append(
                torch.zeros((spec.num_kv_heads, len(layer_mem.spans), cfg.memory.r_max), device=x.device)
            )
            sv_k_list.append(
                torch.zeros((spec.num_kv_heads, len(layer_mem.spans), cfg.memory.r_max), device=x.device)
            )
            sv_v_list.append(
                torch.zeros((spec.num_kv_heads, len(layer_mem.spans), cfg.memory.r_max), device=x.device)
            )

    final_norm = adapter.final_norm(x)
    logits = adapter.lm_head(final_norm).squeeze(0)

    info = {
        "alpha_tilde": torch.stack(alpha_layers, dim=0),  # [L,H,S]
        "z_hard": torch.stack(z_layers, dim=0),
        "gates_k": torch.stack(gates_k_list, dim=0),  # [L,H_kv,S,R]
        "gates_v": torch.stack(gates_v_list, dim=0),
        "sv_k": torch.stack(sv_k_list, dim=0),
        "sv_v": torch.stack(sv_v_list, dim=0),
    }
    return logits, info


def train_step(
    backbone: FrozenBackbone,
    adapter: ArchAdapter,
    modules: SeKVModules,
    input_ids: torch.Tensor,
    cfg: SeKVConfig,
    step: int,
    ste_temperature: float,
) -> dict[str, float]:
    """One optimization step over lightweight SeKV modules only."""
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

    student_logits_rows: list[torch.Tensor] = []
    teacher_logits_rows: list[torch.Tensor] = []
    alpha_rows: list[torch.Tensor] = []
    z_rows: list[torch.Tensor] = []
    y_rows: list[torch.Tensor] = []
    gates_k_rows: list[torch.Tensor] = []
    gates_v_rows: list[torch.Tensor] = []
    sv_k_rows: list[torch.Tensor] = []
    sv_v_rows: list[torch.Tensor] = []

    autocast_enabled = backbone.model.dtype == torch.bfloat16

    for seq in batch_ids:
        with torch.no_grad(), torch.autocast(device_type=seq.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            prefill = backbone.prefill(seq.unsqueeze(0))
            seg = segment_prefill(prefill=prefill, input_ids=seq, cfg=cfg.segmentation)
            span_memory = build_span_memory(prefill=prefill, seg=seg, cfg=cfg, spec=backbone.spec)

        teacher = build_teacher_signals(
            backbone=backbone,
            input_ids=seq,
            prefill=prefill,
            seg=seg,
            cfg=cfg,
        )
        if not teacher.query_positions:
            continue

        with torch.no_grad():
            out_hidden = backbone.model(
                input_ids=seq.unsqueeze(0),
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden0 = out_hidden.hidden_states[0][0].detach().to(dtype=torch.float32)

        for pos in teacher.query_positions:
            logits, info = _student_logits_for_position(
                backbone=backbone,
                adapter=adapter,
                modules=modules,
                span_memory=span_memory,
                position=pos,
                seq_hidden0=hidden0,
                cfg=cfg,
                ste_temperature=ste_temperature,
            )

            student_logits_rows.append(logits)
            teacher_logits_rows.append(teacher.teacher_logits[pos].to(device=logits.device, dtype=torch.float32))

            alpha_rows.append(info["alpha_tilde"])
            z_rows.append(info["z_hard"])
            y_rows.append(teacher.zoom_targets[pos].to(device=logits.device, dtype=torch.float32))

            gates_k_rows.append(info["gates_k"])
            gates_v_rows.append(info["gates_v"])
            sv_k_rows.append(info["sv_k"])
            sv_v_rows.append(info["sv_v"])

    if not student_logits_rows:
        return {
            "loss": 0.0,
            "loss_distill": 0.0,
            "loss_zoom": 0.0,
            "loss_recon": 0.0,
            "loss_budget": 0.0,
            "mean_effective_rank": 0.0,
            "mean_expansion_rate": 0.0,
        }

    student_logits = torch.stack(student_logits_rows, dim=0)
    teacher_logits = torch.stack(teacher_logits_rows, dim=0)

    alpha_all = torch.stack(alpha_rows, dim=0)
    z_all = torch.stack(z_rows, dim=0)
    y_all = torch.stack(y_rows, dim=0)

    gates_k = torch.stack(gates_k_rows, dim=0)
    gates_v = torch.stack(gates_v_rows, dim=0)
    sv_k = torch.stack(sv_k_rows, dim=0)
    sv_v = torch.stack(sv_v_rows, dim=0)

    parts = {
        "distill": distill_loss(student_logits, teacher_logits),
        "zoom": zoom_loss(alpha_all, y_all, w_pos=cfg.loss.w_pos),
        "recon": recon_loss(gates_k, sv_k, gates_v, sv_v),
        "budget": budget_loss(z_all, gates_k, gates_v, beta=cfg.loss.beta),
    }

    budget_weight = budget_anneal(step=step, max_steps=cfg.train.max_steps, cfg=cfg)
    loss, logs = total_loss(parts=parts, weights=cfg.loss, budget_weight=budget_weight)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(modules.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        logs["mean_effective_rank"] = float(effective_rank(gates_k).mean().item())
        logs["mean_expansion_rate"] = float(z_all.mean().item())

    return logs


def _checkpoint_state(modules: SeKVModules, backbone: FrozenBackbone, cfg: SeKVConfig, step: int) -> None:
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": int(step),
        "model_spec": backbone.spec,
        "config": cfg,
        "trainable_parameters": {
            name: tensor.detach().cpu()
            for name, tensor in modules.trainable_parameters().items()
        },
    }
    path = ckpt_dir / f"sekv_step_{step:07d}.pt"
    torch.save(payload, path)


def train(cfg: SeKVConfig) -> None:
    """Train SeKV lightweight modules with streaming data and curriculum schedule."""
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
        pin_memory=True,
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
            _checkpoint_state(modules, backbone, cfg, step + 1)

    _checkpoint_state(modules, backbone, cfg, cfg.train.max_steps)


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
