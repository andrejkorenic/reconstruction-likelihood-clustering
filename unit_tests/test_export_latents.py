"""Tests for analyze.py --export_latents flag."""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch

VAE_DIR = '/home/andrej/Documents/programiranje/new_vae_bundle'


@pytest.fixture(scope="session")
def trained_model_dir(tmp_path_factory):
    """Train a tiny TimeSeries VAE for 2 epochs, return model directory.

    Session-scoped: training runs once and the read-only model directory is
    shared by all tests in this module.
    """
    tmp_path = tmp_path_factory.mktemp("train")

    # Create a minimal TSV: 40 rows, 0 meta cols, 20 timesteps
    n_rows, n_timesteps = 40, 20
    rng = np.random.default_rng(42)
    data = rng.random((n_rows, n_timesteps)).astype(np.float32)
    tsv_path = tmp_path / "test_data.tsv"
    np.savetxt(tsv_path, data, delimiter='\t',
               header='\t'.join(str(i) for i in range(n_timesteps)),
               comments='')

    # Run training via subprocess from tmp_path (pretrained_models/ created there)
    result = subprocess.run(
        [sys.executable, os.path.join(VAE_DIR, 'run.py'),
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
        env={**os.environ, 'PYTHONPATH': VAE_DIR},
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    # Parse MODEL_DIR from stdout
    model_dir = None
    for line in result.stdout.splitlines():
        if line.startswith('MODEL_DIR:'):
            model_dir = line.split('MODEL_DIR:')[1].strip()
            break
    assert model_dir is not None, f"MODEL_DIR not found in stdout:\n{result.stdout}"

    return model_dir, tsv_path, n_rows


@pytest.fixture(scope="session")
def export_result(trained_model_dir, tmp_path_factory):
    """Run --export_latents once and share the result across all tests.

    Session-scoped: the subprocess runs only once for the entire test session.
    """
    model_dir, tsv_path, n_rows = trained_model_dir
    cwd = tmp_path_factory.mktemp("export")
    return subprocess.run(
        [sys.executable, os.path.join(VAE_DIR, 'analyze.py'),
         '--dir', model_dir,
         '--export_latents',
         '--no-cuda'],
        capture_output=True, text=True,
        cwd=str(cwd),
        env={**os.environ, 'PYTHONPATH': VAE_DIR},
    )


def _run_export(model_dir, cwd):
    """Helper: run analyze.py --export_latents."""
    return subprocess.run(
        [sys.executable, os.path.join(VAE_DIR, 'analyze.py'),
         '--dir', model_dir,
         '--export_latents',
         '--no-cuda'],
        capture_output=True, text=True,
        cwd=str(cwd),
        env={**os.environ, 'PYTHONPATH': VAE_DIR},
    )


class TestExportLatents:
    def test_creates_z_mean_npy(self, trained_model_dir, export_result):
        """--export_latents must create z_mean.npy in the model directory."""
        model_dir, tsv_path, n_rows = trained_model_dir
        assert export_result.returncode == 0, f"Export failed:\n{export_result.stderr}"

        z_path = os.path.join(model_dir, 'z_mean.npy')
        assert os.path.exists(z_path), f"z_mean.npy not found at {z_path}"

    def test_z_mean_shape(self, trained_model_dir, export_result):
        """z_mean.npy must have shape (N, z_dim) matching input row count."""
        model_dir, tsv_path, n_rows = trained_model_dir

        z = np.load(os.path.join(model_dir, 'z_mean.npy'))
        assert z.shape[0] == n_rows, f"Expected {n_rows} rows, got {z.shape[0]}"
        assert z.shape[1] == 8, f"Expected z_dim=8, got {z.shape[1]}"

    def test_z_mean_no_nans(self, trained_model_dir, export_result):
        """z_mean must not contain NaN values."""
        model_dir, tsv_path, n_rows = trained_model_dir

        z = np.load(os.path.join(model_dir, 'z_mean.npy'))
        assert not np.isnan(z).any(), "z_mean contains NaN"

    def test_prints_z_mean_path(self, trained_model_dir, export_result):
        """stdout must contain LATENTS_PATH: marker for external parsing."""
        assert 'LATENTS_PATH:' in export_result.stdout
        # Verify the path ends with z_mean.npy
        for line in export_result.stdout.splitlines():
            if line.startswith('LATENTS_PATH:'):
                path = line.split('LATENTS_PATH:')[1].strip()
                assert path.endswith('z_mean.npy')
                break
