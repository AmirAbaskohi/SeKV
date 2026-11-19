from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from sekv.config import LossConfig


def distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Eq.5: KL(p_full || p_SeKV) averaged over sampled query positions."""
    if student_logits.ndim != 2 or student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"student/teacher logits must both be [Q, vocab], got {tuple(student_logits.shape)} and "
            f"{tuple(teacher_logits.shape)}"
        )

    t_log_prob = F.log_softmax(teacher_logits.detach().to(device=student_logits.device, dtype=torch.float32), dim=-1)
    s_log_prob = F.log_softmax(student_logits.to(dtype=torch.float32), dim=-1)
    return F.kl_div(s_log_prob, t_log_prob, log_target=True, reduction="batchmean")


def zoom_loss(alpha_tilde: torch.Tensor, y_target: torch.Tensor, w_pos: float) -> torch.Tensor:
    """Eq.6: BCE between routing probabilities and teacher coverage labels, positives up-weighted by w+.

    Averaged over (position, layer, head, span) entries.
    """
    if alpha_tilde.shape != y_target.shape:
        raise ValueError(f"alpha_tilde and y_target shape mismatch: {tuple(alpha_tilde.shape)} vs {tuple(y_target.shape)}")
    if alpha_tilde.numel() == 0:
        return alpha_tilde.sum() * 0.0

    alpha = alpha_tilde.to(dtype=torch.float32).clamp(1.0e-6, 1.0 - 1.0e-6)
    target = y_target.detach().to(device=alpha.device, dtype=torch.float32)
    logits = torch.log(alpha) - torch.log1p(-alpha)
    pos_weight = torch.tensor([float(w_pos)], device=alpha.device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def recon_loss(
    gates_k: torch.Tensor,
    sv_k: torch.Tensor,
    gates_v: torch.Tensor,
    sv_v: torch.Tensor,
    energy_k: torch.Tensor | None = None,
    energy_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eq.7: gated spectral error sum_i (1 - m_i)^2 sigma_i^2 for keys plus values.

    Inputs are [N, R] over (layer, KV head, span) entries; SVD outputs are constants. With
    energy_* = ||X_S||_F^2 per entry the error is made relative, so spans and heads with large
    key norms do not dominate. Averaged over entries.
    """
    if gates_k.shape != sv_k.shape or gates_v.shape != sv_v.shape:
        raise ValueError("gates and singular values must have matching shapes")
    if gates_k.numel() == 0:
        return gates_k.sum() * 0.0

    def _err(g: torch.Tensor, s: torch.Tensor, e: torch.Tensor | None) -> torch.Tensor:
        s2 = s.detach().to(device=g.device, dtype=torch.float32) ** 2
        err = (((1.0 - g.to(torch.float32)) ** 2) * s2).sum(dim=-1)
        if e is not None:
            err = err / e.detach().to(device=g.device, dtype=torch.float32).clamp_min(1.0e-12)
        return err

    return (_err(gates_k, sv_k, energy_k) + _err(gates_v, sv_v, energy_v)).mean()


def budget_loss(z: torch.Tensor, gates_k: torch.Tensor, gates_v: torch.Tensor, beta: float) -> torch.Tensor:
    """Eq.8: expansion cost z_S plus beta times retained rank sum_i m_i (keys and values).

    z must be the straight-through zoom gate so the first term pushes thresholds up. Both terms
    are averaged over their (position, layer, head, span) / (layer, KV head, span) entries.
    """
    expand_term = z.to(torch.float32).mean() if z.numel() else z.sum() * 0.0
    if gates_k.numel():
        rank_term = gates_k.to(torch.float32).sum(dim=-1).mean() + gates_v.to(torch.float32).sum(dim=-1).mean()
    else:
        rank_term = gates_k.sum() * 0.0
    return expand_term + float(beta) * rank_term


def total_loss(
    parts: dict[str, Any],
    weights: LossConfig,
    budget_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eq.4: L = L_distill + lambda1 L_zoom + lambda2 L_recon + lambda3 L_budget (lambda3 annealed)."""
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
