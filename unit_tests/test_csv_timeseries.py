"""
Tests for csv_timeseries_loader.

Creates a tiny synthetic TSV (3 metadata cols + 10 timesteps, 20 rows)
and verifies shapes, normalisation, and runtime args.
"""

import argparse

import numpy as np
import pytest
import torch

from utils.load_data.data_loader_instances import load_dataset


def make_mock_tsv(tmp_path, n_rows=20, n_timesteps=10):
    """Create a minimal TSV: 3 metadata cols + n_timesteps value cols."""
    header = '\t'.join(
        ['group', 'experiment', 'Selected'] + [str(j + 1) for j in range(n_timesteps)]
    )
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n_rows):
        meta = ['ALS', f'EXP{i:02d}', 'No']
        values = [f'{v:.6f}' for v in rng.uniform(-0.5, 5.0, n_timesteps)]
        rows.append('\t'.join(meta + values))
    content = header + '\n' + '\n'.join(rows)
    tsv_path = tmp_path / 'test_traces.tsv'
    tsv_path.write_text(content)
    return str(tsv_path)


def make_args(csv_path, **overrides):
    defaults = dict(
        dataset_name='csv_timeseries',
        csv_path=csv_path,
        csv_meta_cols=3,       # mock TSV has 3 metadata cols: group, experiment, Selected
        batch_size=8,
        test_batch_size=8,
        training_set_size=0,   # placeholder — overridden by loader
        input_size=[1, 10],
        input_type='continuous',
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


def test_csv_timeseries_split_sizes(tmp_path):
    """80/10/10 split: 20 rows → train=16, val=2, test=2."""
    csv_path = make_mock_tsv(tmp_path, n_rows=20, n_timesteps=10)
    args = make_args(csv_path)
    train_loader, val_loader, test_loader, args = load_dataset(args)

    assert len(train_loader.dataset) == 16, f"Expected 16 train, got {len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 2,   f"Expected 2 val, got {len(val_loader.dataset)}"
    assert len(test_loader.dataset) == 2,  f"Expected 2 test, got {len(test_loader.dataset)}"


def test_csv_timeseries_normalization(tmp_path):
    """All values should be in [0, 1] after MinMax normalization."""
    csv_path = make_mock_tsv(tmp_path, n_rows=20, n_timesteps=10)
    args = make_args(csv_path)
    train_loader, val_loader, test_loader, args = load_dataset(args)

    for loader in [train_loader, val_loader, test_loader]:
        batch = next(iter(loader))
        data = batch[0]  # train is 3-tuple, val/test is 2-tuple; data is first element
        assert data.min().item() >= -1e-5, f"Values below 0: min={data.min().item():.6f}"
        assert data.max().item() <= 1 + 1e-5, f"Values above 1: max={data.max().item():.6f}"


def test_csv_timeseries_runtime_args(tmp_path):
    """Loader must set seq_len=10, input_size=[1,10], input_type='continuous'."""
    csv_path = make_mock_tsv(tmp_path, n_rows=20, n_timesteps=10)
    args = make_args(csv_path)
    _, _, _, args = load_dataset(args)

    assert args.seq_len == 10,                      f"Expected seq_len=10, got {args.seq_len}"
    assert args.input_size == [1, 10],              f"Expected input_size=[1,10], got {args.input_size}"
    assert args.input_type == 'continuous',         f"Expected input_type='continuous', got {args.input_type}"
    assert args.dynamic_binarization is False
    assert args.training_set_size == 16,            f"Expected training_set_size=16, got {args.training_set_size}"


def test_csv_timeseries_no_nan(tmp_path):
    """No NaN values in any batch."""
    csv_path = make_mock_tsv(tmp_path, n_rows=20, n_timesteps=10)
    args = make_args(csv_path)
    train_loader, val_loader, test_loader, args = load_dataset(args)

    for loader in [train_loader, val_loader, test_loader]:
        batch = next(iter(loader))
        data = batch[0]
        assert not torch.isnan(data).any(), "NaN values found in data"


def test_csv_timeseries_data_shape(tmp_path):
    """Train batch shape should be (batch_size, seq_len) = (8, 10)."""
    csv_path = make_mock_tsv(tmp_path, n_rows=20, n_timesteps=10)
    args = make_args(csv_path)
    train_loader, _, _, args = load_dataset(args)

    data, indices, target = next(iter(train_loader))
    assert data.shape == (8, 10), f"Expected (8, 10), got {data.shape}"
    assert indices.shape[1] == 1
    assert target.shape == (8,) or target.shape == (8, 1)


def test_csv_timeseries_via_run_args(tmp_path):
    """load_dataset() with dataset_name='csv_timeseries' must not raise."""
    csv_path = make_mock_tsv(tmp_path, n_rows=30, n_timesteps=15)
    args = make_args(csv_path)
    train_loader, val_loader, test_loader, args = load_dataset(args)
    assert args.seq_len == 15
    assert args.input_size == [1, 15]


def test_csv_timeseries_missing_path(tmp_path):
    """Missing csv_path must raise ValueError."""
    args = make_args(csv_path=None)
    with pytest.raises(ValueError, match="--csv_path is required"):
        load_dataset(args)
