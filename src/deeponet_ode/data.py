"""Synthetic data for the operator (f, k) -> u solving u' = -k u + f, u(0) = 0 on [0, 1].

All functions are pure numpy and deterministic given a numpy Generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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


def grf_factor(cov: np.ndarray) -> np.ndarray:
    """Matrix F with F @ F.T == cov, built from a symmetric eigendecomposition.

    Negative eigenvalues (numerical noise; the RBF kernel on ~1000 points is singular)
    are clipped to zero. Each eigenvector's sign is canonicalized so that its
    largest-magnitude component is positive: LAPACK implementations are free to
    return either sign, and without this step the same seed would give a different
    realization on a different BLAS build.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    pivot = np.abs(eigvecs).argmax(axis=0)
    signs = np.sign(eigvecs[pivot, np.arange(eigvecs.shape[1])])
    signs[signs == 0] = 1.0
    eigvecs = eigvecs * signs
    return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))


def sample_grf(
    rng: np.random.Generator, n: int, grid: np.ndarray, length_scale: float = LENGTH_SCALE
) -> np.ndarray:
    """n samples of a zero-mean, unit-variance Gaussian random field with an RBF kernel."""
    factor = grf_factor(rbf_kernel(grid, grid, length_scale))
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


COLUMNS = ("f_sensors", "k", "t", "u")


def _list_column(arr: np.ndarray) -> pa.ListArray:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, width = arr.shape
    offsets = pa.array(np.arange(0, n * width + 1, width, dtype=np.int32))
    return pa.ListArray.from_arrays(offsets, pa.array(arr.ravel(), type=pa.float32()))


def save_split(split: dict[str, np.ndarray], path: str | Path) -> None:
    """Write a split to Parquet: f_sensors/t/u as list<float32>, k as float32."""
    table = pa.table(
        {
            "f_sensors": _list_column(split["f_sensors"]),
            "k": pa.array(np.asarray(split["k"], dtype=np.float32), type=pa.float32()),
            "t": _list_column(split["t"]),
            "u": _list_column(split["u"]),
        }
    )
    pq.write_table(table, str(path), compression="zstd")


def _list_to_array(column: pa.ChunkedArray) -> np.ndarray:
    col = column.combine_chunks()
    return col.flatten().to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    """Read a Parquet file written by save_split back into float32 arrays."""
    table = pq.read_table(str(path))
    return {
        "f_sensors": _list_to_array(table.column("f_sensors")),
        "k": table.column("k").to_numpy().astype(np.float32),
        "t": _list_to_array(table.column("t")),
        "u": _list_to_array(table.column("u")),
    }
