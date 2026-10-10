import json

import numpy as np
import pytest
import torch

from deeponet_ode.data import make_split, save_split
from deeponet_ode.train import columns_to_tensors, load_data, main


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
