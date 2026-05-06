from __future__ import annotations

import numpy as np
import torch
from scipy.stats import wasserstein_distance


def maximum_mean_discrepancy(
    real: np.ndarray | torch.Tensor,
    generated: np.ndarray | torch.Tensor,
    bandwidth: float | None = None,
) -> float:
    """
    Match the GAN baseline MMD implementation with an RBF kernel and
    a median-heuristic bandwidth when not provided.
    """
    x = torch.tensor(real, dtype=torch.float32) if not isinstance(real, torch.Tensor) else real.to(torch.float32)
    y = torch.tensor(generated, dtype=torch.float32) if not isinstance(generated, torch.Tensor) else generated.to(torch.float32)

    if bandwidth is None:
        n_sub = min(500, x.size(0), y.size(0))
        joint = torch.cat([x[:n_sub], y[:n_sub]], dim=0)
        dists = torch.cdist(joint, joint, p=2)
        triu_idx = torch.triu_indices(dists.size(0), dists.size(1), offset=1)
        bandwidth = dists[triu_idx[0], triu_idx[1]].median().item()
        bandwidth = max(float(bandwidth), 1e-6)

    def rbf_kernel(a: torch.Tensor, b: torch.Tensor, bw: float) -> torch.Tensor:
        a_sq = (a**2).sum(dim=1, keepdim=True)
        b_sq = (b**2).sum(dim=1, keepdim=True)
        dist_sq = a_sq + b_sq.T - 2 * a @ b.T
        return torch.exp(-dist_sq / (2 * bw**2))

    k_xx = rbf_kernel(x, x, bandwidth)
    k_yy = rbf_kernel(y, y, bandwidth)
    k_xy = rbf_kernel(x, y, bandwidth)

    m = x.size(0)
    n = y.size(0)
    mmd = k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2 * k_xy.sum() / (m * n)
    return float(mmd.item())


def sliced_wasserstein_distance(
    real: np.ndarray,
    generated: np.ndarray,
    num_projections: int = 100,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    d = real.shape[1]
    directions = rng.normal(size=(num_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

    distances = []
    for direction in directions:
        real_proj = real @ direction
        generated_proj = generated @ direction
        distances.append(wasserstein_distance(real_proj, generated_proj))

    return float(np.mean(distances))
