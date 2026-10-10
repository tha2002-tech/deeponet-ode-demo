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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push", action="store_true", help="upload --out to the Hub")
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help="dataset repo id, e.g. user/deeponet-ode-data"
    )
    args = parser.parse_args(argv)

    write_dataset(args.out, args.n_train, args.n_test, args.seed, args.repo)
    print(f"Wrote {args.n_train} train and {args.n_test} test rows to {args.out}")
    if args.push:
        push(args.out, args.repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
