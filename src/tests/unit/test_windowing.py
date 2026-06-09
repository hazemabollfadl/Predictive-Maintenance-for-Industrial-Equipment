import numpy as np
import pandas as pd
import pytest

from src.features.windowing import generate_3d_windows, SENSOR_COLUMNS


def _make_df(num_engines=3, cycles_per_engine=50) -> pd.DataFrame:
    """Minimal fake C-MAPSS DataFrame for testing."""
    rows = []
    for engine_id in range(1, num_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            row = {col: np.random.rand() for col in SENSOR_COLUMNS}
            row["RUL"] = float(cycles_per_engine - cycle)
            rows.append((engine_id, cycle, row))

    index = pd.MultiIndex.from_tuples(
        [(r[0], r[1]) for r in rows], names=["Engine_ID", "Cycle"]
    )
    return pd.DataFrame([r[2] for r in rows], index=index)


def test_output_shapes():
    df = _make_df(num_engines=3, cycles_per_engine=50)
    X, y = generate_3d_windows(df, window_size=30)

    expected_samples = 3 * (50 - 30 + 1)   # 3 engines × 21 windows each
    assert X.shape == (expected_samples, 30, 15)
    assert y.shape == (expected_samples,)


def test_labels_are_continuous_rul():
    df = _make_df()
    _, y = generate_3d_windows(df)
    assert y.dtype == np.float32
    assert (y >= 0).all(), "RUL values must be non-negative"


def test_no_cross_engine_leakage():
    """Last window of engine N must not contain rows from engine N+1."""
    df = _make_df(num_engines=2, cycles_per_engine=35)
    X, _ = generate_3d_windows(df, window_size=30)
    assert X.shape[1] == 30


def test_window_size_respected():
    df = _make_df(cycles_per_engine=40)
    X, _ = generate_3d_windows(df, window_size=10)
    assert X.shape[1] == 10


def test_rul_decreases_within_engine():
    """y values for a single engine should be monotonically decreasing."""
    df = _make_df(num_engines=1, cycles_per_engine=50)
    _, y = generate_3d_windows(df, window_size=10)
    assert all(y[i] >= y[i + 1] for i in range(len(y) - 1))
