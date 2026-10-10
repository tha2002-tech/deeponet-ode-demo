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
    for key in (
        "seed", "n_train", "n_test", "n_sensors", "length_scale", "k_range",
        "sensor_grid", "fine_grid_size", "generator", "git_commit",
    ):
        assert key in meta
    assert meta["seed"] == 11 and meta["n_train"] == 8 and meta["fine_grid_size"] == 991

    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "{{" not in card
    assert "| train | 8 |" in card
    assert 'load_dataset("tha2002-tech/deeponet-ode-data")' in card


def test_push_without_token_exits_before_touching_hub(tmp_path, monkeypatch):
    gen = _load_script()
    monkeypatch.setattr(gen, "get_token", lambda: None)

    def boom(*args, **kwargs):
        raise AssertionError("HfApi must not be used without a token")

    monkeypatch.setattr(gen, "HfApi", boom)
    with pytest.raises(SystemExit) as exc:
        gen.main(["--out", str(tmp_path), "--n-train", "2", "--n-test", "1", "--push"])
    assert exc.value.code == 1
