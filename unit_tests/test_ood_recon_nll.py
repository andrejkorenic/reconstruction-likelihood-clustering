"""Tests for analyze.py --ood_recon_nll flag (Phase 6 — Method E)."""
import os
import subprocess
import sys

import numpy as np
import pytest

from pathlib import Path
REPO_ROOT = str(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory):
    """Train a tiny TimeSeries VAE for 2 epochs, return model directory."""
    tmp_path = tmp_path_factory.mktemp("train_recon_nll")

    n_rows, n_timesteps = 40, 20
    rng = np.random.default_rng(42)
    data = rng.random((n_rows, n_timesteps)).astype(np.float32)
    tsv_path = tmp_path / "test_data.tsv"
    np.savetxt(tsv_path, data, delimiter='\t',
               header='\t'.join(str(i) for i in range(n_timesteps)),
               comments='')

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'run.py'),
         '--dataset_name', 'csv_timeseries',
         '--csv_path', str(tsv_path),
         '--csv_meta_cols', '0',
         '--model_name', 'timeseries_vae',
         '--epochs', '2',
         '--warmup', '0',
         '--early_stopping_epochs', '2',
         '--z1_size', '8',
         '--h_size', '64',
         '--batch_size', '16',
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

    return model_dir, tsv_path, n_rows


@pytest.fixture(scope="session")
def ood_nll_result(trained_model_dir, tmp_path_factory):
    """Run --ood_recon_nll once and share the result."""
    model_dir, _, _ = trained_model_dir
    cwd = tmp_path_factory.mktemp("ood_nll_run")
    out_path = cwd / "recon_nll.npy"
    res = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'analyze.py'),
         '--dir', model_dir,
         '--ood_recon_nll', str(out_path),
         '--ood_K', '2',
         '--no-cuda'],
        capture_output=True, text=True,
        cwd=str(cwd),
        env={**os.environ, 'PYTHONPATH': REPO_ROOT},
    )
    return res, out_path


class TestOodReconNll:
    def test_returncode_zero(self, ood_nll_result):
        res, _ = ood_nll_result
        assert res.returncode == 0, f"analyze failed:\n{res.stderr}"

    def test_npy_exists_and_shape(self, ood_nll_result, trained_model_dir):
        _, out_path = ood_nll_result
        _, _, n_rows = trained_model_dir
        assert out_path.exists(), f"{out_path} not written"
        nll = np.load(out_path)
        assert nll.shape == (n_rows,), f"expected ({n_rows},), got {nll.shape}"
        assert nll.dtype == np.float32

    def test_finite_values(self, ood_nll_result):
        _, out_path = ood_nll_result
        nll = np.load(out_path)
        assert np.isfinite(nll).all(), "NLL must be finite"

    def test_prints_marker(self, ood_nll_result):
        res, _ = ood_nll_result
        assert "RECON_NLL_PATH:" in res.stdout
