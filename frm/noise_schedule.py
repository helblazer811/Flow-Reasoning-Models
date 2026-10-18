"""Minimal port of the noise-schedule helpers from upstream `flm/utils.py`.

Only the functions needed by the standalone eval pipeline are included:
  * compute_alpha_exact — ground-truth gamma -> alpha via Gauss-Hermite
  * build_luts          — builds CubicSpline LUTs in both directions
  * alpha_to_gamma      — lookup wrapper for tau -> t
  * gamma_to_alpha      — lookup wrapper for t -> tau

Verbatim ports of the upstream implementations in
experiments/flm_lm1b_upstream_eval/flm/utils.py (lines 480-632), with
unused branches trimmed. No dependency on `flm/`.
"""

from typing import Union

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr


def compute_alpha_exact(
    gamma: np.ndarray,
    K: int,
    n_gh: int = 100,
    sigma_floor: float = 1e-12,
    is_diffusion: bool = False,
) -> np.ndarray:
    """Ground-truth Gamma -> Alpha via Gauss-Hermite integration."""
    gamma = np.asarray(gamma)
    sigma = 1.0 - gamma
    if is_diffusion:
        sigma = np.sqrt(sigma)
    sigma = np.maximum(sigma, sigma_floor)

    m_c = gamma / sigma

    x, w = hermgauss(n_gh)
    w = w / np.sqrt(np.pi)
    z_nodes = np.sqrt(2.0) * x

    m_c_expanded = m_c[:, None]
    z_expanded = z_nodes[None, :]

    L_cu = log_ndtr(z_expanded + m_c_expanded)
    log_prod_c = (K - 1) * L_cu
    q_c = np.sum(w * np.exp(log_prod_c), axis=-1)

    alpha = K / (K - 1.0) * (q_c - 1.0 / K)
    alpha += (gamma - 1) * 1e-10  # ensure monotonicity
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def build_luts(
    K: int, n_points: int = 10000, is_diffusion: bool = False
) -> tuple[CubicSpline, CubicSpline]:
    """Build (alpha->gamma, gamma->alpha) CubicSpline LUTs for vocab size K."""
    gamma_vals = np.linspace(0.0, 1.0, n_points)
    alpha_vals = compute_alpha_exact(gamma_vals, K=K, is_diffusion=is_diffusion)

    lut_g2a = CubicSpline(gamma_vals, alpha_vals)

    sorted_indices = np.argsort(alpha_vals)
    gamma_sorted = gamma_vals[sorted_indices]
    alpha_sorted = alpha_vals[sorted_indices]

    unique_alpha, unique_indices = np.unique(alpha_sorted, return_index=True)
    unique_gamma = gamma_sorted[unique_indices]

    lut_a2g = CubicSpline(unique_alpha, unique_gamma)
    return lut_a2g, lut_g2a


def alpha_to_gamma(
    alpha: Union[np.ndarray, torch.Tensor], lut: CubicSpline
) -> Union[np.ndarray, torch.Tensor]:
    """Alpha -> Gamma lookup. Clipped to [0, 1]."""
    if isinstance(alpha, torch.Tensor):
        dtype = alpha.dtype
        gamma = np.clip(lut(alpha.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(gamma).to(alpha.device, dtype=dtype)
    return np.clip(lut(alpha), 0.0, 1.0)


def gamma_to_alpha(
    gamma: Union[np.ndarray, torch.Tensor], lut: CubicSpline
) -> Union[np.ndarray, torch.Tensor]:
    """Gamma -> Alpha lookup. Clipped to [0, 1]."""
    if isinstance(gamma, torch.Tensor):
        dtype = gamma.dtype
        alpha = np.clip(lut(gamma.cpu().numpy()), 0.0, 1.0)
        return torch.from_numpy(alpha).to(gamma.device, dtype=dtype)
    return np.clip(lut(gamma), 0.0, 1.0)


class TorchLUT:
    """GPU-resident lookup table for fast alpha<->gamma interpolation.

    Eliminates the CPU roundtrip in `alpha_to_gamma` / `gamma_to_alpha`
    by precomputing the LUT values as GPU tensors and doing piecewise
    linear interpolation entirely on-device via `torch.bucketize`.

    Used by training scripts that need the schedule inside the hot loop
    (`experiments/flm_lm1b_reproduction/train_flm.py`,
    `experiments/fmlm_tiny_stories_reproduction/2_train_fmlm_tiny_stories.py`).

    Usage:
        lut_a2g, lut_g2a = build_luts(K=50257)
        gpu_lut = TorchLUT(lut_a2g, device=device)   # build once
        gamma = gpu_lut(tau)                          # fast, stays on GPU
    """

    def __init__(
        self,
        lut: CubicSpline,
        n_points: int = 10_000,
        device: torch.device = None,
    ):
        x = np.linspace(0.0, 1.0, n_points)
        y = np.clip(lut(x), 0.0, 1.0)
        self.x = torch.tensor(x, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

    def __call__(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_f = alpha.float().clamp(0.0, 1.0)
        idx = torch.bucketize(alpha_f, self.x).clamp(0, len(self.x) - 2)
        x0, x1 = self.x[idx], self.x[idx + 1]
        y0, y1 = self.y[idx], self.y[idx + 1]
        t = (alpha_f - x0) / (x1 - x0 + 1e-10)
        return (y0 + t * (y1 - y0)).to(alpha.dtype)
