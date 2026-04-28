"""Tests for analyze.py --export_pseudo_prototypes flag (Phase 7)."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory):
    """Train a tiny TimeSeries VAE for 2 epochs, return model directory."""
    tmp_path = tmp_path_factory.mktemp("train_pseudo_protos")

    n_rows, n_timesteps = 40, 20
    rng = np.random.default_rng(42)
    data = rng.random((n_rows, n_timesteps)).astype(np.float32)
    tsv_path = tmp_path / "test_data.tsv"
    np.savetxt(tsv_path, data, delimiter='\t',
               header='\t'.join(str(i) for i in range(n_timesteps)),
               comments='')

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'run.py'),
         '--dataset_name', 'tabular_timeseries',
         '--ts_path', str(tsv_path),
         '--ts_value_cols', r'^\d+$',
         '--model_name', 'timeseries_vae',
         '--epochs', '2',
         '--warmup', '0',
         '--early_stopping_epochs', '2',
         '--z1_size', '8',
         '--h_size', '64',
         '--batch_size', '16',
         '--number_components', '6',
         '--no-cuda',
         '--seed', '42'],
        capture_output=True, text=True,
        cwd=str(tmp_path),
        env={**os.environ, 'PYTHONPATH': REPO_ROOT},
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    model_dir = None
    for line in result.stdout.splitlines():
        if line.startswith('MODEL_DIR:'):
            model_dir = line.split('MODEL_DIR:')[1].strip()
            break
    assert model_dir is not None, f"MODEL_DIR not found in stdout:\n{result.stdout}"

    return model_dir, tsv_path, n_rows, n_timesteps


@pytest.fixture(scope="session")
def export_result(trained_model_dir, tmp_path_factory):
    """Run --export_pseudo_prototypes once and share the result."""
    model_dir, _, _, _ = trained_model_dir
    cwd = tmp_path_factory.mktemp("export_protos_run")
    out = cwd / "protos.npy"
    res = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'analyze.py'),
         '--dir', model_dir,
         '--export_pseudo_prototypes', str(out),
         '--no-cuda'],
        capture_output=True, text=True,
        cwd=str(cwd),
        env={**os.environ, 'PYTHONPATH': REPO_ROOT},
    )
    return res, out


class TestExportPseudoPrototypes:
    def test_returncode_zero(self, export_result):
        res, _ = export_result
        assert res.returncode == 0, f"analyze failed:\n{res.stderr}"

    def test_npy_shape_and_dtype(self, export_result, trained_model_dir):
        _, out = export_result
        _, _, _, n_timesteps = trained_model_dir
        assert out.exists(), f"{out} not written"
        arr = np.load(out)
        # K=number_components=6 from the trained model
        assert arr.shape == (6, n_timesteps), f"expected (6, {n_timesteps}), got {arr.shape}"
        assert arr.dtype == np.float32

    def test_finite_and_in_unit_interval(self, export_result):
        _, out = export_result
        arr = np.load(out)
        assert np.isfinite(arr).all(), "prototypes must be finite"
        # Beta decoder uses sigmoid → in [0, 1]
        assert (arr >= 0.0).all() and (arr <= 1.0).all(), \
            f"prototypes must be in [0, 1] under Beta decoder; got [{arr.min()}, {arr.max()}]"

    def test_prints_marker(self, export_result):
        res, _ = export_result
        assert "PSEUDO_PROTOTYPES_PATH:" in res.stdout
