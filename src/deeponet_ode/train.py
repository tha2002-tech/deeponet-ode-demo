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

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from deeponet_ode.data import COLUMNS, load_split  # noqa: E402
from deeponet_ode.model import DeepONet, make_branch_input, relative_l2  # noqa: E402

DEFAULT_DATASET = "tha2002-tech/deeponet-ode-data"


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

    def to_cols(split):
        return {key: split[key] for key in COLUMNS}

    return columns_to_tensors(to_cols(ds["train"])), columns_to_tensors(to_cols(ds["test"]))


def evaluate(model: DeepONet, data: dict) -> tuple[float, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        pred = model(make_branch_input(data["f_sensors"], data["k"]), data["t"])
        return relative_l2(pred, data["u"]).item(), pred


def plot_predictions(test: dict, pred: torch.Tensor, path: Path, n_show: int = 3) -> None:
    n_show = min(n_show, len(test["k"]))
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 3.2), squeeze=False)
    t = test["t"][0, :, 0].numpy()
    for i, ax in enumerate(axes[0]):
        ax.plot(t, test["u"][i].numpy(), "k-", label="exact u")
        ax.plot(t, pred[i].numpy(), "r--", label="DeepONet")
        ax.set_title(f"test sample {i}, k={test['k'][i].item():.2f}")
        ax.set_xlabel("t")
    axes[0][0].set_ylabel("u(t)")
    axes[0][0].legend()
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * steps_per_epoch, eta_min=1e-5
    )

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
            print(
                f"epoch {epoch:4d}  train mse {train_mse:.3e}  test rel-L2 {test_rel:.4f}"
                f"  ({time.time() - start:.0f}s)"
            )

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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id")
    parser.add_argument(
        "--local", type=Path, default=None, help="directory with train.parquet/test.parquet"
    )
    parser.add_argument("--epochs", type=int, default=600)
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
