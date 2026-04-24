"""Tests for parquet_timeseries_loader."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from utils.load_data.data_loader_instances import load_dataset


def make_mock_parquet(path: Path, n_rows: int = 20, n_timesteps: int = 10) -> Path:
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({
        "subject_id": [f"S{i:02d}" for i in range(n_rows)],
        "group": np.where(np.arange(n_rows) < n_rows // 2, "ALS", "HCTRL"),
        "roi_idx": np.arange(n_rows, dtype=np.int16) % 5,
        "roi_global_id": [f"S{i:02d}_ROI_{i%5}" for i in range(n_rows)],
        "segment_id": np.int8(1),
        "time_start_sec": np.float32(299.2),
        "time_end_sec": np.float32(599.2),
    })
    values = rng.uniform(-0.5, 5.0, size=(n_rows, n_timesteps)).astype(np.float32)
    time_cols = pd.DataFrame(values, columns=[f"t_{j:03d}" for j in range(n_timesteps)])
    df = pd.concat([meta, time_cols], axis=1)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)
    return path


def make_args(parquet_path: str, **overrides):
    defaults = dict(
        dataset_name="parquet_timeseries",
        parquet_path=parquet_path,
        batch_size=8,
        test_batch_size=8,
        training_set_size=0,
        input_size=[1, 10],
        input_type="continuous",
        dynamic_binarization=False,
        number_components=10,
        pseudoinputs_mean=-0.05,
        pseudoinputs_std=0.01,
        use_training_data_init=False,
        continuous=False,
        use_logit=False,
        lambd=1e-4,
        seed=42,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_parquet_timeseries_split_sizes(tmp_path):
    path = make_mock_parquet(tmp_path / "f.parquet", n_rows=20, n_timesteps=10)
    args = make_args(str(path))
    train, val, test, args = load_dataset(args)
    assert len(train.dataset) == 16
    assert len(val.dataset) == 2
    assert len(test.dataset) == 2
    assert args.seq_len == 10
    assert args.feat_dim == 1
    assert args.input_type == "continuous"


def test_parquet_timeseries_normalization_to_unit_range(tmp_path):
    path = make_mock_parquet(tmp_path / "f.parquet", n_rows=40, n_timesteps=8)
    args = make_args(str(path))
    train, _, _, _ = load_dataset(args)
    # collect all training values; train loader yields 3-tuple (data, indices, target)
    all_vals = np.concatenate([batch[0].numpy().ravel() for batch in train])
    assert all_vals.min() >= 0.0 - 1e-6
    assert all_vals.max() <= 1.0 + 1e-6


def test_parquet_timeseries_ignores_non_sample_t_prefix_cols(tmp_path):
    """time_start_sec / time_end_sec start with 't' but not 't_<digit>', so they
    must NOT land in the sample tensor."""
    path = make_mock_parquet(tmp_path / "f.parquet", n_rows=20, n_timesteps=10)
    args = make_args(str(path))
    _, _, _, args = load_dataset(args)
    # seq_len must be exactly n_timesteps, not n_timesteps + 2
    assert args.seq_len == 10
