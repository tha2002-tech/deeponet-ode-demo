"""Synthetic data for the operator (f, k) -> u solving u' = -k u + f, u(0) = 0 on [0, 1].

All functions are pure numpy and deterministic given a numpy Generator.
"""

from __future__ import annotations

import numpy as np

K_RANGE = (0.1, 2.0)
LENGTH_SCALE = 0.2
N_SENSORS = 100
REFINE = 10


def sensor_grid(m: int = N_SENSORS) -> np.ndarray:
    """m equally spaced sensor/query points on [0, 1] including both endpoints."""
    return np.linspace(0.0, 1.0, m)


def fine_grid(m: int = N_SENSORS, refine: int = REFINE) -> np.ndarray:
    """Fine integration grid with (m-1)*refine + 1 points; sensor_grid(m) == fine_grid(m)[::refine].

    The sensor points are copied in explicitly so the equality holds bit-for-bit
    (linspace of the two sizes otherwise differs by one ulp at some points).
    """
    grid = np.linspace(0.0, 1.0, (m - 1) * refine + 1)
    grid[::refine] = sensor_grid(m)
    return grid


def rbf_kernel(x: np.ndarray, y: np.ndarray, length_scale: float) -> np.ndarray:
    d = x[:, None] - y[None, :]
    return np.exp(-0.5 * (d / length_scale) ** 2)


def sample_grf(
    rng: np.random.Generator, n: int, grid: np.ndarray, length_scale: float = LENGTH_SCALE
) -> np.ndarray:
    """n samples of a zero-mean, unit-variance Gaussian random field with an RBF kernel.

    Uses a symmetric eigendecomposition with negative eigenvalues clipped to zero.
    The RBF kernel on ~1000 points is numerically singular, so Cholesky would need
    a large jitter; clipping gives samples with the same distribution.
    """
    cov = rbf_kernel(grid, grid, length_scale)
    eigvals, eigvecs = np.linalg.eigh(cov)
    factor = eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))
    z = rng.standard_normal((n, grid.size))
    return z @ factor.T


def solve_linear_ode(f_fine: np.ndarray, k: np.ndarray, t_fine: np.ndarray) -> np.ndarray:
    """Exact solution u(t) = int_0^t exp(-k (t - s)) f(s) ds via cumulative trapezoid.

    f_fine: (n, N) samples of f on t_fine; k: (n,); t_fine: (N,). Returns (n, N).
    """
    f_fine = np.asarray(f_fine, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64).reshape(-1, 1)
    t = np.asarray(t_fine, dtype=np.float64)[None, :]
    g = np.exp(k * t) * f_fine
    increments = 0.5 * (g[:, 1:] + g[:, :-1]) * np.diff(t, axis=1)
    integral = np.concatenate(
        [np.zeros((f_fine.shape[0], 1)), np.cumsum(increments, axis=1)], axis=1
    )
    return np.exp(-k * t) * integral


def make_split(
    rng: np.random.Generator,
    n: int,
    m: int = N_SENSORS,
    refine: int = REFINE,
    length_scale: float = LENGTH_SCALE,
    k_range: tuple[float, float] = K_RANGE,
) -> dict[str, np.ndarray]:
    """Generate n (f, k, u) triples sampled at the m-point sensor grid.

    Returns float32 arrays: f_sensors (n, m), k (n,), t (n, m), u (n, m).
    """
    t_fine = fine_grid(m, refine)
    f_fine = sample_grf(rng, n, t_fine, length_scale)
    k = rng.uniform(k_range[0], k_range[1], size=n)
    u_fine = solve_linear_ode(f_fine, k, t_fine)
    sensors = sensor_grid(m)
    return {
        "f_sensors": f_fine[:, ::refine].astype(np.float32),
        "k": k.astype(np.float32),
        "t": np.broadcast_to(sensors, (n, m)).astype(np.float32),
        "u": u_fine[:, ::refine].astype(np.float32),
    }
