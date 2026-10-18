"""Conditional flow matching, fixed-point forcing, and sampling."""

import torch
import torch.nn.functional as F


def corrupt_continuous_with_noise(
    x0: torch.Tensor, t: torch.Tensor, vocab_size: int, noise: torch.Tensor
):
    """Linear interpolation between an *externally drawn* Gaussian noise and one-hot(x0).

        x_t = (1 - t) * noise + t * one_hot(x0)

    Identical to :func:`corrupt_continuous` except the caller supplies ``noise``
    instead of it being drawn internally. This is what lets DPO-style losses share
    the *same* corruption noise across positive/negative/reference forward passes
    (VRPO variance reduction) — see ``methods/dpo_flm.py``.

    Args:
        x0:         (B, L) integer token ids
        t:          (B,)   interpolation coefficients in [0, 1]
        vocab_size: V
        noise:      (B, L, V) externally drawn noise; its dtype drives the output dtype.

    Returns:
        x_t: (B, L, V) corrupted continuous representation
        x0:  (B, L)    original token ids (for cross-entropy in the caller)
    """
    target = F.one_hot(x0, vocab_size).to(dtype=noise.dtype)
    t_view = t.to(noise.dtype).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    x_t = (1.0 - t_view) * noise + t_view * target
    return x_t, x0


def corrupt_continuous(x0: torch.Tensor, t: torch.Tensor, vocab_size: int):
    """Linear interpolation between Gaussian noise and one-hot(x0).

        x_t = (1 - t) * noise + t * one_hot(x0)

    The (B, L, V) tensors are built in bfloat16 to keep memory tractable at
    large vocab sizes. Thin wrapper over :func:`corrupt_continuous_with_noise`
    that draws the noise internally (preserves the original public API).

    Args:
        x0:         (B, L) integer token ids
        t:          (B,)   interpolation coefficients in [0, 1]
        vocab_size: V

    Returns:
        x_t: (B, L, V) corrupted continuous representation (bfloat16)
        x0:  (B, L)    original token ids (for cross-entropy in the caller)
    """
    target = F.one_hot(x0, vocab_size).to(dtype=torch.bfloat16)
    noise = torch.randn_like(target)
    return corrupt_continuous_with_noise(x0, t, vocab_size, noise)


def sample_t_antithetic(
    batch_size: int, device: torch.device, t_min: float = 0.0, t_max: float = 1.0
) -> torch.Tensor:
    """Antithetic time sampling -- lower variance than plain uniform.

    Adapted from ``FLMBase._sample_t_interval`` in david3684/flm.
    """
    n = batch_size
    eps_t = torch.rand(n, device=device)
    offset = torch.arange(n, device=device) / n
    eps_t = (eps_t / n + offset) % 1.0
    perm = torch.randperm(n, device=device)
    eps_t = eps_t[perm]
    return (t_max - t_min) * eps_t + t_min


def held_level_tau(lut_a2g, level, device=None):
    """Model time on the 4,097-point grid whose physical time is closest to ``level``.

    Returns ``(tau, t)``. Used by both held-t training and held-t sampling so that the
    two operate at exactly the same level (``level=0.4`` gives ``tau=647/4096`` for vocab 10).
    """
    grid = torch.linspace(0.0, 1.0, 4097, device=device)
    phys = lut_a2g(grid)
    j = int((phys - float(level)).abs().argmin().item())
    return float(grid[j].item()), float(phys[j].item())


def make_conditional_flm_loss(
    lut_a2g,
    t_min,
    t_max,
    vocab_size,
    self_cond_prob: float = 0.0,
    self_forcing_rollout: float = 0.0,
    sfr_t_min: float = 0.0,
    sfr_k_min: int = 2,
    sfr_k_max: int = 6,
    sfr_cap_value: float = 30.0,
    sfr_noise_scale: float = 0.0,
    sfr_deep_sup_n: int = 1,
    sfr_fixed_depth: int = 0,
    fixed_tau: float = 0.0,
):
    """Build the self-conditioning / detached rollout loss from the public recipe.

    ``fixed_tau > 0`` pins every loss-bearing time and both rollout endpoints to one model
    time. The rollout then has no Euler integration (``dt = 0``); with ``sfr_noise_scale=1``
    each substep is the held-t denoise/re-noise update used at inference.
    """

    def loss_fn(model, x0):
        sol = x0[:, 0]
        mask = x0[:, 1].bool()
        loss_mask = mask
        (B, L) = sol.shape
        if fixed_tau > 0.0:
            tau = torch.full((B,), float(fixed_tau), device=sol.device)
        else:
            tau = sample_t_antithetic(B, sol.device, t_min, t_max)
        t = lut_a2g(tau)
        (x_t, _) = corrupt_continuous(sol, t, vocab_size)
        clean = F.one_hot(sol, vocab_size).to(x_t.dtype)
        x_t = torch.where(mask.unsqueeze(-1), clean, x_t)
        self_cond = None
        if self_cond_prob > 0.0:
            use_sc = (torch.rand(B, device=sol.device) < self_cond_prob).view(B, 1, 1)
            with torch.no_grad():
                sc_logits = model(x_t, tau)
            self_cond = (sc_logits.to(x_t.dtype) * use_sc.to(x_t.dtype)).detach()
        if self_forcing_rollout > 0.0:
            fire = torch.rand(B, device=sol.device) < self_forcing_rollout
            if fire.any():
                if int(sfr_fixed_depth) > 0:
                    k = int(sfr_fixed_depth)
                else:
                    k = int(
                        torch.randint(int(sfr_k_min), int(sfr_k_max) + 1, (1,)).item()
                    )
                if fixed_tau > 0.0:
                    tau_in = torch.full((B,), float(fixed_tau), device=sol.device)
                    tau_start = tau_in
                else:
                    u_in = torch.rand(B, device=sol.device)
                    tau_in = sfr_t_min + (1.0 - sfr_t_min) * u_in
                    u_buf = torch.rand(B, device=sol.device)
                    tau_start = sfr_t_min + tau_in * u_buf
                traj_taus = []
                traj_bufs = []
                with torch.no_grad():
                    t_start = lut_a2g(tau_start)
                    (z, _) = corrupt_continuous(sol, t_start, vocab_size)
                    z = torch.where(mask.unsqueeze(-1), clean, z)
                    sc_prev = None
                    zero_buf = torch.zeros(
                        B, L, vocab_size, dtype=x_t.dtype, device=sol.device
                    )
                    for j in range(k):
                        frac_a = j / k
                        frac_b = (j + 1) / k
                        tau_a = tau_start + (tau_in - tau_start) * frac_a
                        tau_b = tau_start + (tau_in - tau_start) * frac_b
                        traj_taus.append(tau_a)
                        traj_bufs.append(
                            zero_buf
                            if sc_prev is None
                            else sc_prev.to(x_t.dtype).detach()
                        )
                        t_a = lut_a2g(tau_a)
                        t_b = lut_a2g(tau_b)
                        dt = (t_b - t_a).view(-1, 1, 1)
                        raw = model(z, tau_a, self_cond=sc_prev)
                        capped = sfr_cap_value * torch.tanh(raw / sfr_cap_value)
                        logprob = capped - capped.logsumexp(dim=-1, keepdim=True)
                        probs = logprob.exp().to(z.dtype)
                        sc_prev = capped
                        v = (probs - z) / (1.0 - t_a.view(-1, 1, 1) + 1e-05)
                        z_ode = z + dt * v
                        if sfr_noise_scale > 0.0:
                            # Re-noise the clean estimate to the next level with fresh noise.
                            tn = t_b.view(-1, 1, 1)
                            eps = torch.randn_like(z)
                            z_sde = tn * probs + (1.0 - tn) * eps
                            z = (1.0 - sfr_noise_scale) * z_ode + sfr_noise_scale * z_sde
                        else:
                            z = z_ode
                        z = torch.where(mask.unsqueeze(-1), clean, z)
                    traj_taus.append(tau_in)
                    traj_bufs.append(sc_prev.to(x_t.dtype).detach())
                num_states = len(traj_taus)
                n_take = min(int(sfr_deep_sup_n), num_states)
                sel = torch.randperm(num_states, device=sol.device)[:n_take].tolist()
                fm = fire.view(B, 1, 1).to(x_t.dtype)
                fire_f = fire.float()
                n_fire = fire_f.sum().clamp(min=1.0)
                n_fill = (~loss_mask).float().sum(dim=1).clamp(min=1.0)
                base_self_cond = self_cond
                total_loss = z.new_zeros(())
                if (~fire).any():
                    logits_nf = model(x_t, tau, self_cond=base_self_cond)
                    ce_nf = F.cross_entropy(
                        logits_nf.float().view(B * L, vocab_size),
                        sol.reshape(B * L),
                        reduction="none",
                    ).view(B, L)
                    ce_nf = ce_nf.masked_fill(loss_mask, 0.0)
                    per_row_nf = ce_nf.sum(dim=1) / n_fill
                    total_loss = (
                        total_loss + (per_row_nf * (1.0 - fire_f)).sum() / n_fire
                    )
                for s in sel:
                    tau_s = traj_taus[s]
                    buf_s = traj_bufs[s]
                    t_s = lut_a2g(tau_s)
                    (x_t_s, _) = corrupt_continuous(sol, t_s, vocab_size)
                    x_t_s = torch.where(mask.unsqueeze(-1), clean, x_t_s)
                    sc_s = (
                        buf_s
                        if base_self_cond is None
                        else fm * buf_s + (1.0 - fm) * base_self_cond
                    )
                    sc_s = sc_s.detach()
                    logits_s = model(x_t_s, tau_s, self_cond=sc_s)
                    ce_s = F.cross_entropy(
                        logits_s.float().view(B * L, vocab_size),
                        sol.reshape(B * L),
                        reduction="none",
                    ).view(B, L)
                    ce_s = ce_s.masked_fill(loss_mask, 0.0)
                    per_row_s = ce_s.sum(dim=1) / n_fill
                    total_loss = total_loss + (per_row_s * fire_f).sum() / n_fire
                return total_loss
        logits = model(x_t, tau, self_cond=self_cond)
        loss = F.cross_entropy(
            logits.float().view(B * L, vocab_size), sol.reshape(B * L), reduction="none"
        ).view(B, L)
        loss = loss.masked_fill(loss_mask, 0.0)
        return loss.sum() / (~loss_mask).sum().clamp(min=1)

    return loss_fn


@torch.no_grad()
def conditional_generate_samples(
    model,
    sol,
    clue_mask,
    *,
    lut_a2g,
    num_steps,
    vocab_size,
    device=None,
    cap_value=30.0,
    dtype=torch.float32,
    noise_scale=0.0,
    self_cond=False,
    held_tau=None,
    held_iters=256,
    renoise=0.5,
):
    """Generate clue-clamped completions with Euler/SDE or held-level iteration."""
    if device is None:
        device = next(model.parameters()).device
    sol = sol.to(device)
    mask3 = clue_mask.to(device).bool().unsqueeze(-1)
    clue_onehot = F.one_hot(sol, vocab_size).to(dtype)
    (B, L) = sol.shape

    def forward_fn(z, tau, sc=None):
        raw = model(z, tau, self_cond=sc if self_cond else None)
        capped = cap_value * torch.tanh(raw / cap_value)
        return (capped - capped.logsumexp(dim=-1, keepdim=True), capped)

    was_training = model.training
    model.eval()
    try:
        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = torch.randn((B, L, vocab_size), device=device, dtype=dtype)
        z = torch.where(mask3, clue_onehot, z)
        if held_tau is not None:
            (tau_hold, t_h) = held_level_tau(lut_a2g, held_tau, device)
            tau_h = torch.full((B,), tau_hold, device=device, dtype=torch.float32)
            sc_prev = None
            probs = None
            for _ in range(int(held_iters)):
                (logprob, raw) = forward_fn(z, tau_h, sc=sc_prev)
                sc_prev = raw
                probs = logprob.exp()
                eps = torch.randn_like(z)
                z_sde = t_h * probs + (1.0 - t_h) * eps
                z = (1.0 - renoise) * z + renoise * z_sde
                z = torch.where(mask3, clue_onehot, z)
            z = forward_fn(z, tau_h, sc=sc_prev)[0].exp()
            z = torch.where(mask3, clue_onehot, z)
            ids = z.argmax(dim=-1)
            out = (ids,)
            return out if len(out) > 1 else out[0]
        sc_prev = None
        for i in range(num_steps):
            tau_in = tau_vals[i].expand(B)
            t_in = lut_a2g(tau_in)
            t_next = lut_a2g(tau_vals[i + 1].expand(B))
            dt = t_next - t_in
            (logprob, raw) = forward_fn(z, tau_in, sc=sc_prev)
            sc_prev = raw
            probs = logprob.exp()
            v = (probs - z) / (1.0 - t_in.view(-1, 1, 1) + 1e-05)
            z_ode = z + dt.view(-1, 1, 1) * v
            if noise_scale > 0.0:
                tn = t_next.view(-1, 1, 1)
                eps = torch.randn_like(z)
                z_sde = tn * probs + (1.0 - tn) * eps
                z = (1.0 - noise_scale) * z_ode + noise_scale * z_sde
            else:
                z = z_ode
            z = torch.where(mask3, clue_onehot, z)
        z = forward_fn(z, tau_vals[num_steps].expand(B), sc=sc_prev)[0].exp()
        z = torch.where(mask3, clue_onehot, z)
        ids = z.argmax(dim=-1)
        out = (ids,)
        return out if len(out) > 1 else out[0]
    finally:
        if was_training:
            model.train()
