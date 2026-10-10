# DeepONet linear ODE demo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GitHub repository that trains a PyTorch DeepONet on the operator (f, k) ↦ u for `u' = -k u + f, u(0)=0`, pulling its training data from a Hugging Face dataset repository that the same code generates and uploads.

**Architecture:** `data.py` holds pure numpy functions (Gaussian random field sampling, exact solver by variation of constants, split construction, Parquet read/write). `model.py` holds the DeepONet module with input standardization stored as buffers. `train.py` is a CLI that loads data from the Hub or local Parquet, trains, evaluates relative L2, and saves a plot. `scripts/generate_data.py` builds `data/` and optionally pushes it to the Hub together with a dataset card rendered from `dataset_card/README.md`.

**Tech Stack:** Python 3.13 (Anaconda at `C:\Users\ADMIN\anaconda3\python.exe`), torch 2.9.1 CPU, numpy, pyarrow, pandas, matplotlib, `datasets`, `huggingface_hub`, pytest, git, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-10-10-deeponet-ode-design.md`

## Global Constraints

- Repository root: `C:\Users\ADMIN\Downloads\tt\deeponet-ode-demo`. Run every command from there.
- Python interpreter: `C:\Users\ADMIN\anaconda3\python.exe`. Below, `PY` means that path. In Git Bash: `PY=/c/Users/ADMIN/anaconda3/python.exe`.
- Set `KMP_DUPLICATE_LIB_OK=TRUE` in the shell for every Python command (two OpenMP runtimes exist in this environment). In Git Bash: `export KMP_DUPLICATE_LIB_OK=TRUE`.
- Domain t ∈ [0, 1]; m = 100 sensors equally spaced including both endpoints; query grid = sensor grid; k ~ U(0.1, 2.0); GRF RBF kernel length scale 0.2, unit variance; train 2000 rows, test 500 rows; seeds: train `seed`, test `seed + 1`, default seed 0.
- Fine integration grid: 991 points (spacing 1/990) so every sensor point `i/99` lies exactly on it (index `10 i`). This replaces the spec's 1001 so no interpolation is needed.
- Parquet columns and types exactly: `f_sensors list<float32>[100]`, `k float32`, `t list<float32>[100]`, `u list<float32>[100]`.
- Model: branch MLP 101→128→128→64, trunk MLP 1→128→128→64, Tanh, output `einsum('bp,bqp->bq') + bias`.
- GitHub repo id: `tha2002-tech/deeponet-ode-demo`. HF dataset id: `<hf_user>/deeponet-ode-data`, `<hf_user>` taken from `hf auth whoami` in Task 8.
- Commit author is already configured at repo level (`tha2002-tech <tyhamath@gmail.com>`). Every commit message ends with the line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Never commit `data/`, `results/`, `*.pt`.

## Review Focus

1. `generate_data.py --push` with no Hugging Face token must exit with code 1 and a message naming `hf auth login`, before creating anything on the Hub. Test added in Task 5.
2. `train.py --local <missing dir>` or a failing `load_dataset` must exit with a message telling the user to run `scripts/generate_data.py` and use `--local data`. Test added in Task 6.
3. Rows from the Hub arrive as Python lists while local rows arrive as numpy arrays; both must become `(n, 100)` float32 tensors. Test added in Task 6.
4. k = 0 and the k range endpoints 0.1 and 2.0 must give correct solutions from the solver (no overflow from `exp(k t)`, correct limit). Tests added in Task 2.
5. Same seed must reproduce identical arrays and identical Parquet files so the Hub data can be regenerated. Tests added in Task 2 and Task 3.

---

### Task 1: Repository scaffold and environment

**Files:**
- Create: `.gitignore`, `.gitattributes`, `LICENSE`, `requirements.txt`, `pyproject.toml`, `src/deeponet_ode/__init__.py`, `tests/__init__.py`

**Interfaces:**
- Produces: importable package `deeponet_ode`; `pytest` discovers `tests/`.

- [ ] **Step 1: Write the files**

`.gitignore`:
```
__pycache__/
*.pyc
*.pt
*.egg-info/
build/
dist/
data/
results/
.pytest_cache/
```

`.gitattributes`:
```
* text=auto eol=lf
*.png binary
*.parquet binary
```

`LICENSE`: the MIT license text with `Copyright (c) 2026 tha2002-tech`.

`requirements.txt`:
```
torch>=2.2
numpy>=1.26
pandas>=2.0
pyarrow>=14
matplotlib>=3.8
datasets>=2.19
huggingface_hub>=0.23
pytest>=8
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "deeponet-ode"
version = "0.1.0"
description = "DeepONet for the linear first-order ODE u' = -k u + f, with data on the Hugging Face Hub"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
dependencies = [
  "torch>=2.2",
  "numpy>=1.26",
  "pandas>=2.0",
  "pyarrow>=14",
  "matplotlib>=3.8",
  "datasets>=2.19",
  "huggingface_hub>=0.23",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`src/deeponet_ode/__init__.py`:
```python
"""DeepONet for the linear first-order ODE u'(t) = -k u(t) + f(t), u(0) = 0."""

__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

- [ ] **Step 2: Install the missing packages and the package itself**

Run:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE; PY=/c/Users/ADMIN/anaconda3/python.exe
$PY -m pip install datasets huggingface_hub pytest
$PY -m pip install -e . --no-deps
```
Expected: both succeed. `hf` CLI becomes available at `C:\Users\ADMIN\anaconda3\Scripts\hf.exe`.

- [ ] **Step 3: Verify import and empty test run**

Run: `$PY -c "import deeponet_ode, datasets, huggingface_hub; print(deeponet_ode.__version__)"` then `$PY -m pytest -q`
Expected: prints `0.1.0`; pytest reports `no tests ran` with exit code 5.

- [ ] **Step 4: Commit**

```bash
git add .gitignore .gitattributes LICENSE requirements.txt pyproject.toml src tests
git commit -m "Scaffold deeponet_ode package

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Data generation functions (`data.py`)

**Files:**
- Create: `src/deeponet_ode/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `sensor_grid(m: int = 100) -> np.ndarray` shape `(m,)`, float64, `linspace(0, 1, m)`.
  - `fine_grid(m: int = 100, refine: int = 10) -> np.ndarray` shape `((m-1)*refine + 1,)`.
  - `sample_grf(rng: np.random.Generator, n: int, grid: np.ndarray, length_scale: float = 0.2) -> np.ndarray` shape `(n, len(grid))`.
  - `solve_linear_ode(f_fine: np.ndarray, k: np.ndarray, t_fine: np.ndarray) -> np.ndarray` shape `(n, len(t_fine))`.
  - `make_split(rng, n, m=100, refine=10, length_scale=0.2, k_range=(0.1, 2.0)) -> dict[str, np.ndarray]` with keys `f_sensors (n,m) float32`, `k (n,) float32`, `t (n,m) float32`, `u (n,m) float32`.

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:
```python
import numpy as np
import pytest

from deeponet_ode.data import (
    fine_grid,
    make_split,
    sample_grf,
    sensor_grid,
    solve_linear_ode,
)


def test_sensor_grid_endpoints_and_shape():
    g = sensor_grid(100)
    assert g.shape == (100,)
    assert g[0] == 0.0 and g[-1] == 1.0


def test_fine_grid_contains_sensor_grid_exactly():
    g = sensor_grid(100)
    tf = fine_grid(100, refine=10)
    assert tf.shape == (991,)
    np.testing.assert_array_equal(tf[::10], g)


@pytest.mark.parametrize("k", [0.1, 0.7, 2.0])
def test_constant_forcing_matches_closed_form(k):
    tf = fine_grid()
    c = 1.5
    f = np.full((1, tf.size), c)
    u = solve_linear_ode(f, np.array([k]), tf)
    expected = (c / k) * (1.0 - np.exp(-k * tf))
    np.testing.assert_allclose(u[0], expected, atol=1e-5)


def test_zero_k_unit_forcing_gives_identity():
    tf = fine_grid()
    f = np.ones((1, tf.size))
    u = solve_linear_ode(f, np.array([0.0]), tf)
    np.testing.assert_allclose(u[0], tf, atol=1e-12)


def test_solver_initial_condition_is_zero_for_random_f():
    rng = np.random.default_rng(1)
    tf = fine_grid()
    f = sample_grf(rng, 5, tf)
    u = solve_linear_ode(f, rng.uniform(0.1, 2.0, 5), tf)
    assert u.shape == (5, tf.size)
    np.testing.assert_array_equal(u[:, 0], 0.0)


def test_grf_shape_and_unit_variance():
    rng = np.random.default_rng(0)
    tf = fine_grid()
    f = sample_grf(rng, 2000, tf)
    assert f.shape == (2000, 991)
    assert abs(f.std() - 1.0) < 0.1
    assert abs(f.mean()) < 0.1


def test_make_split_shapes_dtypes_and_determinism():
    a = make_split(np.random.default_rng(3), 12)
    b = make_split(np.random.default_rng(3), 12)
    assert a["f_sensors"].shape == (12, 100)
    assert a["k"].shape == (12,)
    assert a["t"].shape == (12, 100)
    assert a["u"].shape == (12, 100)
    for key in ("f_sensors", "k", "t", "u"):
        assert a[key].dtype == np.float32
        np.testing.assert_array_equal(a[key], b[key])
    assert a["k"].min() >= 0.1 and a["k"].max() <= 2.0
    np.testing.assert_array_equal(a["t"][0], sensor_grid(100).astype(np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_data.py -q`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'deeponet_ode.data'`.

- [ ] **Step 3: Write the implementation**

`src/deeponet_ode/data.py`:
```python
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
    """Fine integration grid with (m-1)*refine + 1 points; sensor_grid(m) == fine_grid(m)[::refine]."""
    return np.linspace(0.0, 1.0, (m - 1) * refine + 1)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_data.py -q`
Expected: 9 passed. If `test_grf_shape_and_unit_variance` fails on the std bound, print `f.std()`; a value in 0.9–1.1 is expected, anything else means the eigen-factor is wrong (check `eigvecs * sqrt(eigvals)` multiplies columns, not rows).

- [ ] **Step 5: Commit**

```bash
git add src/deeponet_ode/data.py tests/test_data.py
git commit -m "Add GRF sampling, exact ODE solver and split construction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Parquet read/write (`data.py`)

**Files:**
- Modify: `src/deeponet_ode/data.py` (append)
- Test: `tests/test_data.py` (append)

**Interfaces:**
- Produces:
  - `save_split(split: dict[str, np.ndarray], path: str | Path) -> None` writes Parquet with the exact schema from Global Constraints.
  - `load_split(path: str | Path) -> dict[str, np.ndarray]` returns the same dict layout as `make_split`.
  - `COLUMNS = ("f_sensors", "k", "t", "u")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:
```python
import pyarrow.parquet as pq

from deeponet_ode.data import COLUMNS, load_split, save_split


def test_parquet_round_trip_and_schema(tmp_path):
    split = make_split(np.random.default_rng(5), 7)
    path = tmp_path / "x.parquet"
    save_split(split, path)

    schema = pq.read_schema(path)
    assert schema.names == list(COLUMNS)
    assert str(schema.field("k").type) == "float"
    for name in ("f_sensors", "t", "u"):
        assert str(schema.field(name).type) == "list<item: float>"

    back = load_split(path)
    for key in COLUMNS:
        assert back[key].dtype == np.float32
        np.testing.assert_array_equal(back[key], split[key])


def test_parquet_is_byte_identical_for_same_seed(tmp_path):
    for i in range(2):
        save_split(make_split(np.random.default_rng(9), 5), tmp_path / f"{i}.parquet")
    assert (tmp_path / "0.parquet").read_bytes() == (tmp_path / "1.parquet").read_bytes()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_data.py -q`
Expected: FAIL with `ImportError: cannot import name 'COLUMNS'`.

- [ ] **Step 3: Write the implementation**

Append to `src/deeponet_ode/data.py` (add `from pathlib import Path`, `import pyarrow as pa`, `import pyarrow.parquet as pq` to the imports at the top):
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_data.py -q`
Expected: 11 passed. If the byte-identical test fails, pyarrow is writing a creation timestamp; pass `write_statistics=True` is fine but check for `store_schema`/metadata differences by comparing `pq.read_metadata(...)`; as a fallback write with `pq.write_table(table, path, compression="zstd", data_page_version="1.0")`.

- [ ] **Step 5: Commit**

```bash
git add src/deeponet_ode/data.py tests/test_data.py
git commit -m "Add Parquet save/load for dataset splits

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: DeepONet model (`model.py`)

**Files:**
- Create: `src/deeponet_ode/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces:
  - `class DeepONet(nn.Module)` with `__init__(self, n_sensors=100, branch_extra=1, width=128, n_hidden=2, p=64)`, buffers `branch_mean (n_sensors+branch_extra,)` and `branch_std (n_sensors+branch_extra,)`, method `set_branch_stats(mean: Tensor, std: Tensor) -> None`, and `forward(branch_in: Tensor (B, n_sensors+branch_extra), t: Tensor (B, Q, 1)) -> Tensor (B, Q)`.
  - `make_branch_input(f_sensors: Tensor (B, m), k: Tensor (B,)) -> Tensor (B, m+1)` concatenating k as the last column.
  - `relative_l2(pred: Tensor (B, Q), target: Tensor (B, Q)) -> Tensor scalar` = mean over B of `||pred-target|| / ||target||`.

- [ ] **Step 1: Write the failing tests**

`tests/test_model.py`:
```python
import torch

from deeponet_ode.model import DeepONet, make_branch_input, relative_l2


def test_forward_shape_and_finite():
    model = DeepONet()
    f = torch.randn(4, 100)
    k = torch.rand(4)
    t = torch.linspace(0, 1, 100).view(1, 100, 1).expand(4, 100, 1)
    out = model(make_branch_input(f, k), t)
    assert out.shape == (4, 100)
    assert torch.isfinite(out).all()


def test_make_branch_input_puts_k_last():
    f = torch.zeros(3, 5)
    k = torch.tensor([1.0, 2.0, 3.0])
    x = make_branch_input(f, k)
    assert x.shape == (3, 6)
    assert torch.equal(x[:, -1], k)


def test_branch_stats_are_saved_in_state_dict_and_applied():
    model = DeepONet(n_sensors=5)
    mean = torch.arange(6.0)
    std = torch.full((6,), 2.0)
    model.set_branch_stats(mean, std)
    sd = model.state_dict()
    assert torch.equal(sd["branch_mean"], mean)
    assert torch.equal(sd["branch_std"], std)
    fresh = DeepONet(n_sensors=5)
    fresh.load_state_dict(sd)
    x = torch.randn(2, 6)
    t = torch.rand(2, 3, 1)
    assert torch.allclose(fresh(x, t), model(x, t))


def test_relative_l2_values():
    a = torch.tensor([[3.0, 4.0]])
    assert relative_l2(a, a).item() == 0.0
    assert abs(relative_l2(torch.zeros(1, 2), a).item() - 1.0) < 1e-6


def test_short_training_reduces_loss():
    torch.manual_seed(0)
    model = DeepONet(n_sensors=10, width=32, p=16)
    f = torch.randn(16, 10)
    k = torch.rand(16)
    t = torch.linspace(0, 1, 20).view(1, 20, 1).expand(16, 20, 1)
    target = torch.sin(3 * t[..., 0]) * f[:, :1]
    x = make_branch_input(f, k)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = torch.mean((model(x, t) - target) ** 2)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.5 * first
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_model.py -q`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'deeponet_ode.model'`.

- [ ] **Step 3: Write the implementation**

`src/deeponet_ode/model.py`:
```python
"""DeepONet (Lu et al. 2021): branch net on the input function, trunk net on the query point."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def mlp(in_dim: int, width: int, n_hidden: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """G(f, k)(t) ≈ sum_p branch_p([f; k]) * trunk_p(t) + b.

    Branch input is standardized with `branch_mean`/`branch_std`, stored as buffers so
    they travel with the state_dict.
    """

    def __init__(
        self,
        n_sensors: int = 100,
        branch_extra: int = 1,
        width: int = 128,
        n_hidden: int = 2,
        p: int = 64,
    ) -> None:
        super().__init__()
        branch_dim = n_sensors + branch_extra
        self.branch = mlp(branch_dim, width, n_hidden, p)
        self.trunk = mlp(1, width, n_hidden, p)
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("branch_mean", torch.zeros(branch_dim))
        self.register_buffer("branch_std", torch.ones(branch_dim))

    def set_branch_stats(self, mean: Tensor, std: Tensor) -> None:
        self.branch_mean.copy_(mean)
        self.branch_std.copy_(std.clamp_min(1e-8))

    def forward(self, branch_in: Tensor, t: Tensor) -> Tensor:
        x = (branch_in - self.branch_mean) / self.branch_std
        b = self.branch(x)  # (B, p)
        tr = self.trunk(t)  # (B, Q, p)
        return torch.einsum("bp,bqp->bq", b, tr) + self.bias


def make_branch_input(f_sensors: Tensor, k: Tensor) -> Tensor:
    """Concatenate sensor values and k into the branch input (B, m + 1)."""
    return torch.cat([f_sensors, k.reshape(-1, 1)], dim=1)


def relative_l2(pred: Tensor, target: Tensor) -> Tensor:
    """Mean over the batch of ||pred - target||_2 / ||target||_2."""
    num = torch.linalg.vector_norm(pred - target, dim=1)
    den = torch.linalg.vector_norm(target, dim=1).clamp_min(1e-12)
    return (num / den).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_model.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/deeponet_ode/model.py tests/test_model.py
git commit -m "Add DeepONet model with stored input standardization

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Data generation script and dataset card

**Files:**
- Create: `scripts/generate_data.py`, `dataset_card/README.md`
- Test: `tests/test_generate_data.py`

**Interfaces:**
- Consumes: `make_split`, `save_split` from Task 2–3.
- Produces: CLI `python scripts/generate_data.py [--out data] [--n-train 2000] [--n-test 500] [--seed 0] [--push] [--repo <id>]`, with `main(argv: list[str] | None = None) -> int`. Writes `train.parquet`, `test.parquet`, `metadata.json`, `README.md` into `--out`. Template placeholders in `dataset_card/README.md`: `{{HF_REPO}}`, `{{GIT_COMMIT}}`, `{{N_TRAIN}}`, `{{N_TEST}}`, `{{SEED}}`.
- Defaults: `--repo` defaults to the string `HF_USER/deeponet-ode-data`; Task 8 replaces `HF_USER` with the real user name.

- [ ] **Step 1: Write the dataset card template**

`dataset_card/README.md`:
````markdown
---
license: mit
pretty_name: DeepONet linear ODE data
task_categories:
- tabular-regression
tags:
- deeponet
- operator-learning
- ode
- physics
- synthetic
size_categories:
- n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
  - split: test
    path: test.parquet
---

# DeepONet linear ODE data

Synthetic training data for learning the solution operator of the linear
first-order ODE

```
u'(t) = -k · u(t) + f(t),   u(0) = 0,   t ∈ [0, 1]
```

with a DeepONet. The input is a pair (f, k): `f` is a random smooth forcing
function and `k ∈ [0.1, 2]` is a damping constant. The target is the exact
solution `u(t) = ∫₀ᵗ e^{-k (t-s)} f(s) ds`.

Code that generates this dataset and trains a DeepONet on it:
**https://github.com/tha2002-tech/deeponet-ode-demo**

## Splits

| split | rows |
|-------|------|
| train | {{N_TRAIN}} |
| test  | {{N_TEST}} |

## Schema

| column      | type            | length | meaning                                      |
|-------------|-----------------|--------|----------------------------------------------|
| `f_sensors` | list<float32>   | 100    | f evaluated at 100 equally spaced sensors on [0, 1] |
| `k`         | float32         | 1      | damping constant, uniform in [0.1, 2.0]       |
| `t`         | list<float32>   | 100    | query times (same 100-point grid as the sensors) |
| `u`         | list<float32>   | 100    | exact solution at `t`                         |

## Generation

- `f` is sampled from a zero-mean Gaussian random field on [0, 1] with an RBF
  kernel of length scale 0.2 and unit variance, on a 991-point grid.
- `u` is computed by the variation-of-constants integral with the cumulative
  trapezoid rule on that grid (error about 1e-6), then subsampled to the
  100-point sensor grid.
- Seed {{SEED}} for train, seed {{SEED}}+1 for test.
- Produced by `scripts/generate_data.py` at commit `{{GIT_COMMIT}}` of the
  repository above. `metadata.json` in this repo records all parameters.

## Usage

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("{{HF_REPO}}")
f = np.array(ds["train"]["f_sensors"], dtype=np.float32)  # (N, 100)
k = np.array(ds["train"]["k"], dtype=np.float32)          # (N,)
u = np.array(ds["train"]["u"], dtype=np.float32)          # (N, 100)
```

## Reference

Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G. E. (2021). Learning
nonlinear operators via DeepONet based on the universal approximation theorem
of operators. *Nature Machine Intelligence* 3, 218–229.
````

- [ ] **Step 2: Write the failing tests**

`tests/test_generate_data.py`:
```python
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from deeponet_ode.data import load_split

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("generate_data", ROOT / "scripts" / "generate_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_writes_all_files_with_expected_shapes(tmp_path):
    gen = _load_script()
    rc = gen.main(["--out", str(tmp_path), "--n-train", "8", "--n-test", "3", "--seed", "11"])
    assert rc == 0
    train = load_split(tmp_path / "train.parquet")
    test = load_split(tmp_path / "test.parquet")
    assert train["u"].shape == (8, 100)
    assert test["u"].shape == (3, 100)
    assert not np.array_equal(train["k"][:3], test["k"])

    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    for key in ("seed", "n_train", "n_test", "n_sensors", "length_scale", "k_range", "sensor_grid", "fine_grid_size", "generator", "git_commit"):
        assert key in meta
    assert meta["seed"] == 11 and meta["n_train"] == 8 and meta["fine_grid_size"] == 991

    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "{{" not in card
    assert "| train | 8 |" in card
    assert "load_dataset(\"HF_USER/deeponet-ode-data\")" in card


def test_push_without_token_exits_before_touching_hub(tmp_path, monkeypatch):
    gen = _load_script()
    monkeypatch.setattr(gen, "get_token", lambda: None)

    def boom(*args, **kwargs):
        raise AssertionError("HfApi must not be used without a token")

    monkeypatch.setattr(gen, "HfApi", boom)
    with pytest.raises(SystemExit) as exc:
        gen.main(["--out", str(tmp_path), "--n-train", "2", "--n-test", "1", "--push"])
    assert exc.value.code == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_generate_data.py -q`
Expected: FAIL with `FileNotFoundError` for `scripts/generate_data.py`.

- [ ] **Step 4: Write the script**

`scripts/generate_data.py`:
```python
"""Generate the DeepONet linear-ODE dataset and optionally push it to the Hugging Face Hub.

Usage:
    python scripts/generate_data.py                       # writes data/
    python scripts/generate_data.py --push --repo USER/deeponet-ode-data
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, get_token

from deeponet_ode import data as D

ROOT = Path(__file__).resolve().parents[1]
CARD_TEMPLATE = ROOT / "dataset_card" / "README.md"
DEFAULT_REPO = "HF_USER/deeponet-ode-data"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render_card(repo: str, n_train: int, n_test: int, seed: int) -> str:
    text = CARD_TEMPLATE.read_text(encoding="utf-8")
    for key, value in {
        "HF_REPO": repo,
        "GIT_COMMIT": git_commit(),
        "N_TRAIN": str(n_train),
        "N_TEST": str(n_test),
        "SEED": str(seed),
    }.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def write_dataset(out: Path, n_train: int, n_test: int, seed: int, repo: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    train = D.make_split(np.random.default_rng(seed), n_train)
    test = D.make_split(np.random.default_rng(seed + 1), n_test)
    D.save_split(train, out / "train.parquet")
    D.save_split(test, out / "test.parquet")

    meta = {
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        "n_sensors": D.N_SENSORS,
        "n_query": D.N_SENSORS,
        "length_scale": D.LENGTH_SCALE,
        "k_range": list(D.K_RANGE),
        "sensor_grid": D.sensor_grid().tolist(),
        "fine_grid_size": int(D.fine_grid().size),
        "generator": "scripts/generate_data.py",
        "github": "https://github.com/tha2002-tech/deeponet-ode-demo",
        "git_commit": git_commit(),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "README.md").write_text(render_card(repo, n_train, n_test, seed), encoding="utf-8")


def push(out: Path, repo: str) -> None:
    if get_token() is None:
        print(
            "No Hugging Face token found. Run `hf auth login` with a write token, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Regenerate data from {git_commit()[:8]}",
    )
    print(f"Pushed {out} to https://huggingface.co/datasets/{repo}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push", action="store_true", help="upload --out to the Hub")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="dataset repo id, e.g. user/deeponet-ode-data")
    args = parser.parse_args(argv)

    write_dataset(args.out, args.n_train, args.n_test, args.seed, args.repo)
    print(f"Wrote {args.n_train} train and {args.n_test} test rows to {args.out}")
    if args.push:
        push(args.out, args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_generate_data.py -q`
Expected: 2 passed.

- [ ] **Step 6: Generate the real local dataset and inspect it**

Run:
```bash
$PY scripts/generate_data.py
$PY -c "from deeponet_ode.data import load_split; d=load_split('data/train.parquet'); print(d['u'].shape, d['k'].min(), d['k'].max(), abs(d['u']).max())"
ls -la data
```
Expected: `(2000, 100) ~0.1 ~2.0 <~3`; `train.parquet` roughly 1.5–2.5 MB, `test.parquet` roughly 0.4–0.6 MB.

- [ ] **Step 7: Commit (data/ stays untracked)**

```bash
git add scripts/generate_data.py dataset_card/README.md tests/test_generate_data.py
git status --short   # must not list data/
git commit -m "Add dataset generator with Hub upload and dataset card template

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Training CLI (`train.py`)

**Files:**
- Create: `src/deeponet_ode/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `load_split`, `COLUMNS` (Task 3); `DeepONet`, `make_branch_input`, `relative_l2` (Task 4).
- Produces:
  - `columns_to_tensors(cols: dict[str, Sequence]) -> dict[str, Tensor]` accepting numpy arrays or Python lists; returns float32 tensors `f_sensors (n,100)`, `k (n,)`, `t (n,100,1)`, `u (n,100)`.
  - `load_data(dataset: str | None, local: Path | None) -> tuple[dict, dict]` (train, test).
  - `run_training(train, test, epochs, batch_size, lr, seed, out: Path) -> dict` returning metrics and writing `model.pt`, `metrics.json`, `predictions.png` to `out`.
  - `main(argv=None) -> int`; CLI flags `--dataset` (default `HF_USER/deeponet-ode-data`), `--local`, `--epochs 300`, `--batch-size 64`, `--lr 1e-3`, `--seed 0`, `--out results`.

- [ ] **Step 1: Write the failing tests**

`tests/test_train.py`:
```python
import json

import numpy as np
import pytest
import torch

from deeponet_ode.data import make_split, save_split
from deeponet_ode.train import columns_to_tensors, load_data, main, run_training


def test_columns_to_tensors_accepts_lists_and_arrays():
    split = make_split(np.random.default_rng(0), 3)
    as_lists = {key: value.tolist() for key, value in split.items()}
    for source in (split, as_lists):
        tensors = columns_to_tensors(source)
        assert tensors["f_sensors"].shape == (3, 100)
        assert tensors["k"].shape == (3,)
        assert tensors["t"].shape == (3, 100, 1)
        assert tensors["u"].shape == (3, 100)
        assert all(v.dtype == torch.float32 for v in tensors.values())


def test_missing_local_dir_exits_with_hint(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_data(None, tmp_path / "nope")
    assert "generate_data" in str(exc.value)


def test_run_training_writes_outputs(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    save_split(make_split(np.random.default_rng(1), 32), data_dir / "train.parquet")
    save_split(make_split(np.random.default_rng(2), 8), data_dir / "test.parquet")
    out = tmp_path / "results"
    rc = main(["--local", str(data_dir), "--epochs", "2", "--batch-size", "16", "--out", str(out)])
    assert rc == 0
    assert (out / "model.pt").exists()
    assert (out / "predictions.png").exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) >= {"test_rel_l2", "train_mse", "epochs", "n_train", "n_test"}
    assert metrics["epochs"] == 2 and metrics["n_train"] == 32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_train.py -q`
Expected: FAIL at import with `ModuleNotFoundError: No module named 'deeponet_ode.train'`.

- [ ] **Step 3: Write the implementation**

`src/deeponet_ode/train.py`:
```python
"""Train a DeepONet on the linear ODE dataset.

    python -m deeponet_ode.train                      # data from the Hugging Face Hub
    python -m deeponet_ode.train --local data         # data from scripts/generate_data.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # two OpenMP runtimes ship with conda numpy + torch

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from deeponet_ode.data import COLUMNS, load_split
from deeponet_ode.model import DeepONet, make_branch_input, relative_l2

DEFAULT_DATASET = "HF_USER/deeponet-ode-data"


def columns_to_tensors(cols: dict[str, Sequence]) -> dict[str, torch.Tensor]:
    """Convert columnar data (numpy arrays or nested lists) to float32 tensors."""
    out = {key: torch.as_tensor(np.asarray(cols[key], dtype=np.float32)) for key in COLUMNS}
    out["t"] = out["t"].unsqueeze(-1)
    return out


def load_data(dataset: str | None, local: Path | None) -> tuple[dict, dict]:
    if local is not None:
        train_path, test_path = Path(local) / "train.parquet", Path(local) / "test.parquet"
        if not (train_path.exists() and test_path.exists()):
            raise SystemExit(
                f"{local} has no train.parquet/test.parquet. Run `python scripts/generate_data.py` "
                "and pass `--local data`."
            )
        return columns_to_tensors(load_split(train_path)), columns_to_tensors(load_split(test_path))

    from datasets import load_dataset

    try:
        ds = load_dataset(dataset)
    except Exception as err:  # network, auth, wrong id
        raise SystemExit(
            f"Could not load '{dataset}' from the Hugging Face Hub ({type(err).__name__}: {err}). "
            "Run `python scripts/generate_data.py` and pass `--local data` instead."
        ) from err
    to_cols = lambda split: {key: split[key] for key in COLUMNS}  # noqa: E731
    return columns_to_tensors(to_cols(ds["train"])), columns_to_tensors(to_cols(ds["test"]))


def evaluate(model: DeepONet, data: dict) -> tuple[float, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        pred = model(make_branch_input(data["f_sensors"], data["k"]), data["t"])
        return relative_l2(pred, data["u"]).item(), pred


def plot_predictions(test: dict, pred: torch.Tensor, path: Path, n_show: int = 3) -> None:
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3.2), sharey=False)
    t = test["t"][0, :, 0].numpy()
    for i, ax in enumerate(axes):
        ax.plot(t, test["u"][i].numpy(), "k-", label="exact u")
        ax.plot(t, pred[i].numpy(), "r--", label="DeepONet")
        ax.set_title(f"test sample {i}, k={test['k'][i].item():.2f}")
        ax.set_xlabel("t")
    axes[0].set_ylabel("u(t)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_training(
    train: dict, test: dict, epochs: int, batch_size: int, lr: float, seed: int, out: Path
) -> dict:
    torch.manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)

    branch_train = make_branch_input(train["f_sensors"], train["k"])
    model = DeepONet(n_sensors=train["f_sensors"].shape[1])
    model.set_branch_stats(branch_train.mean(0), branch_train.std(0))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = max(1, math.ceil(len(branch_train) / batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch, eta_min=1e-5)

    n = len(branch_train)
    start = time.time()
    train_mse = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(branch_train[idx], train["t"][idx])
            loss = torch.mean((pred - train["u"][idx]) ** 2)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item() * len(idx)
        train_mse = running / n
        if epoch % 10 == 0 or epoch == epochs:
            test_rel, _ = evaluate(model, test)
            print(f"epoch {epoch:4d}  train mse {train_mse:.3e}  test rel-L2 {test_rel:.4f}  ({time.time() - start:.0f}s)")

    test_rel, pred = evaluate(model, test)
    torch.save(model.state_dict(), out / "model.pt")
    plot_predictions(test, pred, out / "predictions.png")
    metrics = {
        "test_rel_l2": test_rel,
        "train_mse": train_mse,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "n_train": int(n),
        "n_test": int(len(test["k"])),
        "seconds": time.time() - start,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"final test rel-L2 {test_rel:.4f}; wrote {out / 'model.pt'}, {out / 'predictions.png'}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id")
    parser.add_argument("--local", type=Path, default=None, help="directory with train.parquet/test.parquet")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    train, test = load_data(args.dataset, args.local)
    run_training(train, test, args.epochs, args.batch_size, args.lr, args.seed, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest -q`
Expected: 21 passed (all tasks so far).

- [ ] **Step 5: Full local training run**

Run: `$PY -m deeponet_ode.train --local data`
Expected: finishes in a few minutes on CPU; final `test rel-L2` below 0.02. If it is above 0.02, first try `--epochs 600`; if still above, check that `branch_std` for the k column is not near zero and that `t` has shape `(B, 100, 1)`. Record the final number for the README.

- [ ] **Step 6: Copy the plot into assets and commit**

```bash
mkdir -p assets && cp results/predictions.png assets/predictions.png
git add src/deeponet_ode/train.py tests/test_train.py assets/predictions.png
git commit -m "Add training CLI with Hub/local loading, evaluation and plot

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Repository README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: metrics from Task 6 Step 5 (fill in the measured rel-L2 and training time).
- Placeholder `HF_USER` remains until Task 8.

- [ ] **Step 1: Write README.md**

````markdown
# DeepONet for a linear first-order ODE

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/HF_USER/deeponet-ode-data)
![tests](https://img.shields.io/badge/tests-pytest-blue)

A compact PyTorch example of operator learning. A DeepONet learns the map
(f, k) ↦ u for

```
u'(t) = -k · u(t) + f(t),   u(0) = 0,   t ∈ [0, 1]
```

where `f` is a random smooth forcing function and `k ∈ [0.1, 2]` is a damping
constant. The training data lives on the Hugging Face Hub at
[`HF_USER/deeponet-ode-data`](https://huggingface.co/datasets/HF_USER/deeponet-ode-data)
and is generated by `scripts/generate_data.py` in this repository.

![predictions](assets/predictions.png)

## Results

| metric | value |
|--------|-------|
| test relative L2 error | REL_L2 |
| training time (CPU) | SECONDS s |
| train / test functions | 2000 / 500 |

## Quick start

```bash
pip install -e .
python -m deeponet_ode.train            # downloads HF_USER/deeponet-ode-data and trains
```

Outputs go to `results/`: `model.pt`, `metrics.json`, `predictions.png`.

Regenerate the data locally and train on it instead:

```bash
python scripts/generate_data.py          # writes data/
python -m deeponet_ode.train --local data
```

Push a regenerated dataset to your own Hub account:

```bash
hf auth login
python scripts/generate_data.py --push --repo YOUR_USER/deeponet-ode-data
```

## How it works

- **Data** (`src/deeponet_ode/data.py`): `f` is sampled from a Gaussian random
  field (RBF kernel, length scale 0.2) on a 991-point grid; `u` is the exact
  variation-of-constants integral computed by the cumulative trapezoid rule;
  both are stored at 100 equally spaced points. Splits are Parquet files with
  columns `f_sensors`, `k`, `t`, `u`.
- **Model** (`src/deeponet_ode/model.py`): branch MLP 101→128→128→64 on
  `[f_sensors, k]`, trunk MLP 1→128→128→64 on `t`, output is their dot product
  plus a bias. Input standardization statistics are stored in the checkpoint.
- **Training** (`src/deeponet_ode/train.py`): Adam, MSE, cosine learning-rate
  decay, 300 epochs with batches of 64 functions × 100 query points.

## Dataset ↔ code link

- The dataset card on the Hub names this repository, the generator script and
  the git commit that produced the files; `metadata.json` there records every
  parameter and the sensor grid.
- `train.py` defaults to the Hub dataset id, so a fresh clone trains with no
  manual data step.

## Development

```bash
pip install -r requirements.txt
pytest
```

Troubleshooting on Windows + conda: if you see `OMP: Error #15` about
`libiomp5md.dll`, set `KMP_DUPLICATE_LIB_OK=TRUE` (the training script sets it
for itself).

## Reference

Lu, L., Jin, P., Pang, G., Zhang, Z., Karniadakis, G. E. (2021). Learning
nonlinear operators via DeepONet based on the universal approximation theorem
of operators. *Nature Machine Intelligence* 3, 218–229.

## License

MIT
````

Replace `REL_L2` and `SECONDS` with the values from `results/metrics.json` (format rel-L2 as a percentage with two decimals, seconds as an integer).

- [ ] **Step 2: Check that no placeholder other than HF_USER remains**

Run: `grep -n "REL_L2\|SECONDS\|{{" README.md`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Add README with results, usage and dataset link

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Hugging Face login, upload, and wiring the real dataset id

**Files:**
- Modify: `README.md`, `scripts/generate_data.py` (DEFAULT_REPO), `src/deeponet_ode/train.py` (DEFAULT_DATASET), `tests/test_generate_data.py` (expected card string)

**Interfaces:**
- Consumes: `hf` CLI from Task 1; generator from Task 5.
- Produces: public dataset `https://huggingface.co/datasets/<hf_user>/deeponet-ode-data`.

- [ ] **Step 1: User action — log in to Hugging Face**

Ask the user to run, in their own terminal, with a **write** token from https://huggingface.co/settings/tokens:
```bash
C:\Users\ADMIN\anaconda3\Scripts\hf.exe auth login
```
Wait for them to confirm. Do not enter the token yourself.

- [ ] **Step 2: Read the user name**

Run: `/c/Users/ADMIN/anaconda3/Scripts/hf.exe auth whoami`
Expected: prints the user name. Call it `HFU` below.

- [ ] **Step 3: Replace the placeholder everywhere**

```bash
grep -rl "HF_USER" README.md scripts src tests | xargs sed -i "s/HF_USER/HFU/g"
grep -rn "HF_USER" . --exclude-dir=.git ; echo "exit $?"   # expect: no matches, exit 1
```

- [ ] **Step 4: Re-run tests**

Run: `$PY -m pytest -q`
Expected: 21 passed (the card assertion in `test_generate_data.py` now checks `HFU/deeponet-ode-data`).

- [ ] **Step 5: Commit the id change, then generate and push**

```bash
git add -A
git commit -m "Point dataset id at HFU/deeponet-ode-data

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
$PY scripts/generate_data.py --push
```
Expected: `Pushed ... to https://huggingface.co/datasets/HFU/deeponet-ode-data`. The commit happens first so the card's `git_commit` points at a commit that will exist on GitHub.

- [ ] **Step 6: Verify the Hub copy loads and trains**

```bash
$PY -c "from datasets import load_dataset; ds=load_dataset('HFU/deeponet-ode-data'); print(ds); print(len(ds['train'][0]['f_sensors']))"
$PY -m deeponet_ode.train --epochs 20 --out results_hub
```
Expected: `DatasetDict` with train 2000 rows, test 500 rows, 4 columns; `100`; the training run starts and prints epochs (20 epochs is only a smoke test of the Hub path). Open `https://huggingface.co/datasets/HFU/deeponet-ode-data` in the browser pane and confirm the card renders, the dataset viewer shows the four columns, and the GitHub link is present.

- [ ] **Step 7: Clean up**

```bash
rm -rf results_hub
git status --short   # expect clean
```

---

### Task 9: Create the GitHub repository and push

**Files:** none new.

**Interfaces:**
- Consumes: `gh` (installed, user logged in).

- [ ] **Step 1: Confirm gh auth**

Run: `"/c/Program Files/GitHub CLI/gh.exe" auth status`
Expected: `Logged in to github.com account tha2002-tech`.

- [ ] **Step 2: Create the public repo from the local one and push**

```bash
"/c/Program Files/GitHub CLI/gh.exe" repo create tha2002-tech/deeponet-ode-demo --public --source . --remote origin --push --description "DeepONet for the linear ODE u' = -k u + f; data on the Hugging Face Hub"
git remote -v
git log --oneline origin/main | head -3
```
Expected: repo URL printed; `origin` points at `https://github.com/tha2002-tech/deeponet-ode-demo.git`; `origin/main` matches local `main`.

- [ ] **Step 3: Verify both directions of the link**

Open in the browser pane:
- `https://github.com/tha2002-tech/deeponet-ode-demo` — README renders, badge links to the HF dataset, `assets/predictions.png` shows.
- `https://huggingface.co/datasets/HFU/deeponet-ode-data` — card links to the GitHub repo and the commit hash in the card exists in `git log`.

Run: `git log --oneline | grep -c .` and compare the hash in the card (`curl -s https://huggingface.co/datasets/HFU/deeponet-ode-data/raw/main/metadata.json | grep git_commit`) against `git log --format=%H`.
Expected: the hash is in the log.

- [ ] **Step 4: Report**

Tell the user both URLs, the measured test error, and how to retrain or regenerate.
