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
