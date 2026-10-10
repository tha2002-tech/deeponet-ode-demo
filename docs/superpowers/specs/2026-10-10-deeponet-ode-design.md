# DeepONet for a linear first-order ODE — design

Date: 2026-10-10
Status: approved in conversation, awaiting written review

## 1. Goal

Build a small, reproducible example that learns the solution operator of the
linear first-order ODE

    u'(t) = -k * u(t) + f(t),   u(0) = 0,   t in [0, 1]

with a DeepONet (branch net + trunk net) in PyTorch. The operator maps the pair
(f, k) to the solution u. The input function f is a random smooth function and
k is a scalar damping constant.

Two public artifacts are produced and linked to each other:

- GitHub repository `tha2002-tech/deeponet-ode-demo`: source code, tests,
  data-generation script, training script, README.
- Hugging Face dataset repository `<hf_user>/deeponet-ode-data`: the generated
  train/test data in Parquet plus a dataset card that links back to the GitHub
  repository. `<hf_user>` is filled in from `hf auth whoami` once the user has
  logged in.

Success criteria:

- `python scripts/generate_data.py` regenerates the dataset bit-for-bit from a
  fixed seed and can push it to the Hub.
- `python -m deeponet_ode.train` downloads the dataset from the Hub by default,
  trains on CPU in a few minutes, and reports a test relative L2 error below
  about 2 percent, saving a plot of predictions versus ground truth.
- Both READMEs link to each other; the HF dataset card names the exact script
  and commit that produced the data.
- `pytest` passes.

## 2. Non-goals

- Nonlinear ODEs, PDEs, multiple dimensions.
- GPU support beyond whatever `torch` picks up automatically.
- Hyper-parameter search, experiment tracking, model upload to the Hub.
- Packaging to PyPI.

## 3. Mathematical setup

Exact solution by variation of constants:

    u(t) = integral_0^t exp(-k (t - s)) f(s) ds

Input functions f are samples of a zero-mean Gaussian random field on [0, 1]
with an RBF (squared-exponential) kernel, length scale 0.2, unit variance,
evaluated on a fine grid of 1001 points. The integral is evaluated on that fine
grid with the cumulative trapezoid rule; this is accurate to roughly 1e-6 for
these smooth f and is treated as ground truth.

k is drawn uniformly from [0.1, 2.0].

DeepONet inputs and outputs:

- Branch input: f sampled at m = 100 equally spaced sensor points on [0, 1]
  (including both endpoints), concatenated with k. Dimension 101.
- Trunk input: a query time t. Dimension 1.
- Output: u(t) approximated as the dot product of the two latent vectors plus
  a learned scalar bias.

## 4. Data

### 4.1 Generation (`src/deeponet_ode/data.py`)

Pure functions, seedable, numpy only:

- `sensor_grid(m=100)` -> array of shape (m,) on [0, 1].
- `sample_grf(rng, n, grid, length_scale=0.2)` -> (n, len(grid)). Cholesky of the
  RBF kernel with a small jitter on the diagonal.
- `solve_linear_ode(f_fine, k, t_fine)` -> u on the fine grid, via cumulative
  trapezoid of `exp(k*s) f(s)` then multiplied by `exp(-k t)`. Vectorized over
  samples.
- `make_split(rng, n, m=100, n_query=100)` -> dict of arrays:
  `f_sensors (n, m)`, `k (n,)`, `t (n, n_query)`, `u (n, n_query)`. Query points
  are the same 100-point grid as the sensors so that every row is a full
  trajectory; this keeps the storage layout simple and lets the plot script show
  whole curves.

### 4.2 Files and schema

`scripts/generate_data.py` writes to `data/`:

- `train.parquet` (2000 rows) and `test.parquet` (500 rows). Columns:

  | column      | type            | shape | meaning                        |
  |-------------|-----------------|-------|--------------------------------|
  | f_sensors   | list<float32>   | 100   | f at the sensor grid           |
  | k           | float32         | 1     | damping constant               |
  | t           | list<float32>   | 100   | query times                    |
  | u           | list<float32>   | 100   | exact solution at `t`          |

- `metadata.json`: seed, m, n_query, length scale, k range, sensor grid, fine
  grid size, generator script path, git commit hash if available.
- `README.md` dataset card, copied from `dataset_card/README.md` with the
  `<hf_user>` and commit placeholders filled in.

Total size is roughly 3 MB.

### 4.3 Upload

`scripts/generate_data.py --push --repo <hf_user>/deeponet-ode-data` creates the
dataset repo if missing and uploads the contents of `data/` with
`huggingface_hub.HfApi.upload_folder`. Without `--push` it only writes locally.
Login is the user's responsibility (`hf auth login`); the script fails with a
clear message if no token is found.

### 4.4 Dataset card (`dataset_card/README.md`)

YAML front matter: `license: mit`, `task_categories: [tabular-regression]`,
`tags: [deeponet, operator-learning, ode, physics]`, `size_categories: n<10K`,
and a `configs` block pointing `train` and `test` at the two Parquet files so
`load_dataset` works without a loading script. Body: problem statement,
schema table, generation procedure, link to the GitHub repository and to
`scripts/generate_data.py`, citation of Lu et al. 2021 (DeepONet).

## 5. Model (`src/deeponet_ode/model.py`)

```
class DeepONet(nn.Module):
    def __init__(self, n_sensors=100, branch_extra=1, width=128, depth=3, p=64)
    def forward(self, branch_in: (B, n_sensors+branch_extra), t: (B, Q, 1)) -> (B, Q)
```

- Branch: MLP 101 -> 128 -> 128 -> 64, Tanh activations.
- Trunk: MLP 1 -> 128 -> 128 -> 64, Tanh activations.
- Output: `einsum('bp,bqp->bq', branch, trunk) + bias`.
- Inputs are standardized with statistics computed from the training split
  (f mean/std, k mean/std); the statistics are saved with the checkpoint.

## 6. Training (`src/deeponet_ode/train.py`)

CLI (argparse):

```
python -m deeponet_ode.train [--dataset <hf_user>/deeponet-ode-data | --local data/]
                             [--epochs 300] [--batch-size 64] [--lr 1e-3]
                             [--seed 0] [--out results/]
```

- Loads data with `datasets.load_dataset(name)` or from local Parquet via
  pandas; converts to float32 tensors.
- Adam, MSE loss over all query points in the batch, cosine LR decay to 1e-5 over the run.
- Every 10 epochs logs train MSE and test relative L2 error
  `||u_pred - u||_2 / ||u||_2` averaged over test functions.
- At the end writes `results/model.pt`, `results/metrics.json`, and
  `results/predictions.png` with three test trajectories (true vs predicted).
- `results/` and `data/` are git-ignored; one `predictions.png` is committed
  under `assets/` for the README.

## 7. Repository layout

```
deeponet-ode-demo/
  README.md                    overview, badges, HF link, how to reproduce
  LICENSE                      MIT
  requirements.txt             torch, numpy, pandas, pyarrow, datasets,
                               huggingface_hub, matplotlib, pytest
  pyproject.toml               minimal, makes `src/` importable
  .gitignore                   data/, results/, __pycache__, *.pt
  assets/predictions.png
  dataset_card/README.md
  scripts/generate_data.py
  src/deeponet_ode/__init__.py
  src/deeponet_ode/data.py
  src/deeponet_ode/model.py
  src/deeponet_ode/train.py
  tests/test_data.py
  tests/test_model.py
  docs/superpowers/specs/2026-10-10-deeponet-ode-design.md
```

## 8. Linking the two repositories

- GitHub README: a Hugging Face dataset badge and link, a code snippet showing
  `load_dataset("<hf_user>/deeponet-ode-data")`, and the generation command.
- HF dataset card: link to the GitHub repository, the generator script path,
  and the commit hash embedded by `generate_data.py`.
- `train.py` default `--dataset` value is the HF repo id, so cloning the GitHub
  repository and running training pulls data from the Hub with no manual step.

## 9. Error handling

- Missing HF token on `--push`: exit with message pointing to `hf auth login`.
- `load_dataset` failure (offline, wrong id): message suggests `--local data/`
  after running the generator.
- Cholesky failure in GRF sampling: jitter 1e-8 on the diagonal; raise if it
  still fails rather than silently changing the kernel.

## 10. Testing

- `tests/test_data.py`
  - For constant f ≡ c the solver matches `(c/k)(1 - exp(-k t))` to 1e-5.
  - For k = 0 and f ≡ 1 the solver returns u = t.
  - `make_split` returns the documented shapes and dtypes; same seed gives
    identical arrays.
  - Parquet round trip preserves shapes and values.
- `tests/test_model.py`
  - Forward pass returns shape (B, Q) and is finite.
  - A few hundred Adam steps on one tiny synthetic batch reduce the loss
    (smoke test that gradients flow).
- End-to-end check (manual, not in pytest): full training reaches the error
  target in section 1.

## 11. Environment and operational steps

- Python: Anaconda `C:\Users\ADMIN\anaconda3\python.exe` (3.13, torch 2.9.1
  CPU already present). Install `datasets huggingface_hub pytest`. Set
  `KMP_DUPLICATE_LIB_OK=TRUE` for this environment because two OpenMP runtimes
  are present; documented in the README troubleshooting section.
- git identity set at repository level: `tha2002-tech <tyhamath@gmail.com>`.
- GitHub CLI installed via winget; the user runs `gh auth login`, then the
  assistant creates the public repository and pushes.
- Hugging Face: the user runs `hf auth login` with a write token; the assistant
  then runs the generator with `--push`.

## 12. Order of work

1. Scaffold repository, `.gitignore`, `requirements.txt`, `pyproject.toml`.
2. `data.py` with tests (TDD).
3. `model.py` with tests (TDD).
4. `generate_data.py`, produce `data/` locally, verify schema.
5. `train.py`, run training locally against `--local data/`, check error target,
   save plot.
6. Dataset card and README.
7. User logs in to HF; push dataset; verify `load_dataset` from the Hub.
8. Switch training default to the Hub id and re-run once.
9. User logs in to gh; create GitHub repo, push; verify links both ways.
