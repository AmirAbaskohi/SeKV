from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from sekv.config import LossConfig


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL(p_full || p_student) averaged over mined positions (temperature 1)."""
    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError(
            "student_logits and teacher_logits must both be rank-2 [Q, vocab], got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits shape mismatch: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )

    t_logits = teacher_logits.detach().to(device=student_logits.device, dtype=torch.float32)
    s_logits = student_logits.to(dtype=torch.float32)

    teacher_prob = F.softmax(t_logits, dim=-1)
    teacher_log_prob = torch.log(teacher_prob.clamp_min(1.0e-12))
    student_log_prob = F.log_softmax(s_logits, dim=-1)

    kl_per_q = torch.sum(teacher_prob * (teacher_log_prob - student_log_prob), dim=-1)
    return kl_per_q.mean()


def zoom_loss(alpha_tilde: torch.Tensor, y_target: torch.Tensor, w_pos: float) -> torch.Tensor:
    """Positive-weighted BCE between routing probabilities and fixed teacher coverage labels."""
    if alpha_tilde.shape != y_target.shape:
        raise ValueError(
            f"alpha_tilde and y_target shape mismatch: {tuple(alpha_tilde.shape)} vs {tuple(y_target.shape)}"
        )

    alpha = alpha_tilde.to(dtype=torch.float32)
    target = y_target.detach().to(device=alpha.device, dtype=torch.float32)

    alpha = alpha.clamp_min(1.0e-6).clamp_max(1.0 - 1.0e-6)
    logits = torch.log(alpha) - torch.log(1.0 - alpha)

    pos_weight = torch.tensor([float(w_pos)], device=alpha.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def recon_loss(
    gates_k: torch.Tensor,
    sv_k: torch.Tensor,
    gates_v: torch.Tensor,
    sv_v: torch.Tensor,
) -> torch.Tensor:
    """Gated spectral error summed across keys+values, heads, layers, and spans."""
    if gates_k.shape != sv_k.shape or gates_v.shape != sv_v.shape:
        raise ValueError(
            "gates and singular values must have matching shapes for each side: "
            f"gates_k {tuple(gates_k.shape)} sv_k {tuple(sv_k.shape)}; "
            f"gates_v {tuple(gates_v.shape)} sv_v {tuple(sv_v.shape)}"
        )

    gk = gates_k.to(dtype=torch.float32)
    gv = gates_v.to(dtype=torch.float32)
    sk = sv_k.detach().to(device=gk.device, dtype=torch.float32)
    sv = sv_v.detach().to(device=gv.device, dtype=torch.float32)

    err_k = ((1.0 - gk) ** 2) * (sk ** 2)
    err_v = ((1.0 - gv) ** 2) * (sv ** 2)
    return err_k.sum() + err_v.sum()


def budget_loss(
    z_hard: torch.Tensor,
    gates_k: torch.Tensor,
    gates_v: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Expansion + rank budget regularizer with keys and values both counted in rank term."""
    z = z_hard.to(dtype=torch.float32)
    gk = gates_k.to(dtype=torch.float32)
    gv = gates_v.to(dtype=torch.float32)

    expand_term = z.sum()
    rank_term = gk.sum() + gv.sum()
    return expand_term + float(beta) * rank_term


def total_loss(
    parts: dict[str, Any],
    weights: LossConfig,
    budget_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted total objective with annealed budget coefficient."""
    required = {"distill", "zoom", "recon", "budget"}
    missing = sorted(required.difference(parts.keys()))
    if missing:
        raise KeyError(f"Missing loss part(s): {', '.join(missing)}")

    l_distill = parts["distill"].to(dtype=torch.float32)
    l_zoom = parts["zoom"].to(dtype=torch.float32)
    l_recon = parts["recon"].to(dtype=torch.float32)
    l_budget = parts["budget"].to(dtype=torch.float32)

    total = (
        l_distill
        + float(weights.lambda_zoom) * l_zoom
        + float(weights.lambda_recon) * l_recon
        + float(budget_weight) * l_budget
    )

    logs = {
        "loss": float(total.detach().item()),
        "loss_distill": float(l_distill.detach().item()),
        "loss_zoom": float(l_zoom.detach().item()),
        "loss_recon": float(l_recon.detach().item()),
        "loss_budget": float(l_budget.detach().item()),
        "budget_weight": float(budget_weight),
    }
    return total, logs
