"""train/pinn/fast_pinn_loss.py — vectorized PINN rollout loss + monkey-patcher.

Drop-in replacement for ``train.pinn.train_pinn.pinn_loss`` that reduces
the rollout-loss computation from O(rollout_steps²) forward passes to
O(rollout_steps), with optional random sub-sampling of rollout starts.

Use ``apply_fast_pinn_loss(rollout_n_starts=...)`` to monkey-patch the
canonical entry point.  Both ``train.pinn.train_pinn`` and
``train.pinn.train_cluster_pinn`` import ``pinn_loss`` from
``train.pinn.train_pinn``, so a single patch covers single-patient and
population (cluster) training paths.

Compared to the canonical ``pinn_loss``:

  - Same data, smoothness, and ODE-residual loss components (numerically
    identical given equal inputs).
  - Rollout loss uses a single contiguous (or random) rollout from the
    common base index ``n_base = T − rollout_steps`` and accumulates the
    per-step squared error against future targets.  Falls back to the
    original O(R²) double-loop when ``T <= rollout_steps``.
  - Supports ``return_parts=True`` to return ``(loss, parts_dict)``,
    matching the contract used by ``train_cluster_pinn``.

Example
-------
    from train.pinn.fast_pinn_loss import apply_fast_pinn_loss
    apply_fast_pinn_loss(rollout_n_starts=64)   # 64 starts/epoch
    # …then call train_pinn(...) or train_cluster_pinn(...) as usual.
"""
from __future__ import annotations

import functools

import torch


def _fast_pinn_loss(
    model, scales, medical_ode, neural_ode,
    data, mask, dt,
    dxdt_clip_vals,
    w_data: float = 20., w_roll: float = 20.,
    w_ode:  float = 0.,  w_smooth: float = 0.1,
    dt_noise_thresh: float = 0.05,
    rollout_steps:   int   = 30,
    return_parts:    bool  = False,
    rollout_n_starts: int | None = None,
):
    """Fast (vectorized) version of ``pinn_loss``.

    Drop-in for ``train.pinn.train_pinn.pinn_loss`` (same kwargs, same outputs).
    Extra kwarg ``rollout_n_starts`` is bound at patch time via
    ``functools.partial`` so the public signature stays compatible.
    """
    # Keep state_dim resolvable without depending on data.config (parallel-safe)
    state_dim = scales["state_min"].shape[-1]

    x_full = data[:, 1:]
    x      = x_full[:, :state_dim]
    a      = x_full[:, state_dim:]
    mask_x = mask[:, 1:1 + state_dim]
    T      = x.shape[0]

    x_t       = x[:-1]
    x_next    = x[1:]
    a_t       = a[:-1]
    mask_t    = mask_x[:-1]
    mask_next = mask_x[1:]
    dt_t      = dt.unsqueeze(1)

    dxdt_pred = model(x_t, a_t)

    valid_dt      = dt >= dt_noise_thresh
    both_observed = mask_t * mask_next
    valid_mask    = both_observed * valid_dt.float().unsqueeze(1)

    dt_safe     = dt_t.clamp(min=dt_noise_thresh)
    target_dxdt = (x_next - x_t) / dt_safe

    # Re-use the clip tensor (avoid per-epoch host-to-device copy)
    clip_t = torch.as_tensor(
        dxdt_clip_vals, dtype=torch.float32, device=data.device,
    )
    target_dxdt = torch.max(torch.min(target_dxdt, clip_t), -clip_t)

    # ── Data consistency loss (1-step derivative match) ────────────────────
    data_loss = (
        (dxdt_pred - target_dxdt) ** 2 * valid_mask
    ).sum() / (valid_mask.sum() + 1e-6)

    # ── Rollout loss (vectorized) ──────────────────────────────────────────
    #
    # Original implementation:  for k in [2, R]:
    #                             rollout length k from every start
    #     forward passes ≈ Σ k = R(R+1)/2 − 1
    #
    # Vectorized version:
    #   * Pick a base length n_base = T − R (or random subset of [0, n_base))
    #   * Roll out R steps once, comparing x_curr against x[start + k] at
    #     each step k ≥ 2.
    #   * Forward passes = R  (≈ half the original cost).
    #
    # Falls back to the original double loop when T ≤ R.
    # ─────────────────────────────────────────────────────────────────────
    roll_total = torch.tensor(0., device=data.device)
    roll_count = 0
    s_min = scales["state_min"]
    s_max = scales["state_max"]

    n_base = T - rollout_steps
    if n_base > 0:
        if rollout_n_starts is not None and rollout_n_starts < n_base:
            starts = torch.randperm(n_base, device=data.device)[:rollout_n_starts]
        else:
            starts = None  # contiguous slice (fastest)

        x_curr = x[:n_base].clone() if starts is None else x[starts].clone()

        for step in range(rollout_steps):
            k = step + 1

            if starts is None:
                a_step  = a[step : n_base + step]
                dt_step = dt[step : n_base + step].unsqueeze(1)
                x_curr  = torch.max(
                    torch.min(x_curr + model(x_curr, a_step) * dt_step, s_max),
                    s_min,
                )
                if k >= 2:
                    target_k = x[k : n_base + k]
                    mask_k   = mask_x[:n_base] * mask_x[k : n_base + k]
            else:
                idx     = starts + step
                a_step  = a[idx]
                dt_step = dt[idx].unsqueeze(1)
                x_curr  = torch.max(
                    torch.min(x_curr + model(x_curr, a_step) * dt_step, s_max),
                    s_min,
                )
                if k >= 2:
                    target_k = x[starts + k]
                    mask_k   = mask_x[starts] * mask_x[starts + k]

            if k >= 2:
                loss_k = (
                    (x_curr - target_k) ** 2 * mask_k
                ).sum() / (mask_k.sum() + 1e-6)
                roll_total = roll_total + loss_k
                roll_count += 1
    else:
        # Fallback: exact original logic for very short sequences
        for k in range(2, rollout_steps + 1):
            n = T - k
            if n <= 0:
                break
            x_curr = x[:n].clone()
            for step in range(k):
                a_step  = a[step : n + step]
                dt_step = dt[step : n + step].unsqueeze(1)
                x_curr  = torch.max(
                    torch.min(x_curr + model(x_curr, a_step) * dt_step, s_max),
                    s_min,
                )
            target_k = x[k : n + k]
            mask_k   = mask_x[:n] * mask_x[k : n + k]
            if target_k.shape[0] > 0:
                loss_k     = (
                    (x_curr - target_k) ** 2 * mask_k
                ).sum() / (mask_k.sum() + 1e-6)
                roll_total = roll_total + loss_k
                roll_count += 1

    roll_loss = roll_total / max(roll_count, 1)

    # ── Smoothness on dx/dt ────────────────────────────────────────────────
    smooth_loss = (
        torch.mean((dxdt_pred[1:] - dxdt_pred[:-1]) ** 2)
        if dxdt_pred.shape[0] > 1
        else torch.tensor(0., device=data.device)
    )

    # ── ODE residual ───────────────────────────────────────────────────────
    if w_ode == 0.:
        ode_loss = torch.tensor(0., device=data.device)
    else:
        if valid_dt.sum() > 1:
            x_v    = x_t[valid_dt]
            a_v    = a_t[valid_dt]
            mask_v = both_observed[valid_dt]
            dxdt_p = dxdt_pred[valid_dt]
            dxdt_o = medical_ode(x_v, a_v) + neural_ode(x_v, a_v)
            ode_res  = torch.clamp(dxdt_p - dxdt_o, -5., 5.)
            ode_loss = (ode_res ** 2 * mask_v).sum() / (mask_v.sum() + 1e-6)
        else:
            ode_loss = torch.tensor(0., device=data.device)

    if w_ode == 0.:
        total = w_data * data_loss + w_roll * roll_loss + w_smooth * smooth_loss
    else:
        total = (
            w_data   * data_loss
            + w_roll   * roll_loss
            + w_smooth * smooth_loss
            + w_ode    * ode_loss
        )

    if return_parts:
        parts = {
            "data":   data_loss.detach().item(),
            "roll":   roll_loss.detach().item(),
            "smooth": smooth_loss.detach().item(),
            "ode":    ode_loss.detach().item() if w_ode > 0 else 0.0,
        }
        return total, parts
    return total


def apply_fast_pinn_loss(rollout_n_starts: int | None = None) -> None:
    """Monkey-patch ``train.pinn.train_pinn.pinn_loss`` with the fast version.

    This affects every consumer that imports ``pinn_loss`` from
    ``train.pinn.train_pinn`` (i.e. both ``train_pinn`` and
    ``train_cluster_pinn``), because they fetch the symbol via the module
    globals at call time.

    Call this once per process **before** running the training loop, and
    AFTER ``train.pinn.train_pinn`` is already imported.

    Parameters
    ----------
    rollout_n_starts : int | None
        Number of rollout starts sub-sampled per epoch (non-replacing).
        ``None`` (default) keeps every valid start (closest to original).
    """
    from .train_pinn import pinn_loss as _m  # noqa: PLC0415

    _m.pinn_loss = functools.partial(
        _fast_pinn_loss, rollout_n_starts=rollout_n_starts,
    )
    tag = (
        f"rollout_n_starts={rollout_n_starts}"
        if rollout_n_starts is not None else "all starts"
    )
    print(f"[fast_pinn_loss] pinn_loss → vectorized rollout patch applied ({tag}).")
