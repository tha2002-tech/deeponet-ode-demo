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


import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from deeponet_ode.data import COLUMNS, load_split, save_split  # noqa: E402


def test_parquet_round_trip_and_schema(tmp_path):
    split = make_split(np.random.default_rng(5), 7)
    path = tmp_path / "x.parquet"
    save_split(split, path)

    schema = pq.read_schema(path)
    assert schema.names == list(COLUMNS)
    assert schema.field("k").type == pa.float32()
    for name in ("f_sensors", "t", "u"):
        assert pa.types.is_list(schema.field(name).type)
        assert schema.field(name).type.value_type == pa.float32()

    back = load_split(path)
    for key in COLUMNS:
        assert back[key].dtype == np.float32
        np.testing.assert_array_equal(back[key], split[key])


def test_parquet_is_byte_identical_for_same_seed(tmp_path):
    for i in range(2):
        save_split(make_split(np.random.default_rng(9), 5), tmp_path / f"{i}.parquet")
    assert (tmp_path / "0.parquet").read_bytes() == (tmp_path / "1.parquet").read_bytes()


from deeponet_ode.data import grf_factor, rbf_kernel  # noqa: E402


def test_grf_factor_has_canonical_eigenvector_signs():
    """eigh eigenvector signs differ between BLAS builds; the factor must not depend on them."""
    tf = fine_grid()
    factor = grf_factor(rbf_kernel(tf, tf, 0.2))
    assert factor.shape == (tf.size, tf.size)
    idx = np.abs(factor).argmax(axis=0)
    assert (factor[idx, np.arange(factor.shape[1])] >= 0).all()
    np.testing.assert_allclose(factor @ factor.T, rbf_kernel(tf, tf, 0.2), atol=1e-10)
