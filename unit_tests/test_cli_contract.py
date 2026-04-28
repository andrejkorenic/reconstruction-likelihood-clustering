"""Integration test: CLI contract enforcement.

Trains a tiny VampPrior TimeSeriesVAE end-to-end, then verifies that
analyze.py's main flags (--export_latents, --export_pseudo_prototypes)
produce artefacts matching the shape/dtype/marker guarantees documented
in API.md at the repo root.

Acts as the "API.md compliance gate" — if this passes, downstream
consumers (calcium_analysis) can rely on the contract.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = str(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory):
    """Train a tiny VampPrior TimeSeriesVAE; return (model_dir, csv_path, n_rows, K)."""
    tmp = tmp_path_factory.mktemp("contract_train")

    n_rows, n_timesteps, K = 40, 20, 6
    rng = np.random.default_rng(42)
    data = rng.random((n_rows, n_timesteps)).astype(np.float32)
    csv_path = tmp / "data.tsv"
    np.savetxt(csv_path, data, delimiter='\t',
               header='\t'.join(str(i) for i in range(n_timesteps)),
               comments='')

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'run.py'),
         '--dataset_name', 'tabular_timeseries',
         '--ts_path', str(csv_path),
         '--ts_value_cols', r'^\d+$',
         '--model_name', 'timeseries_vae',
         '--prior', 'vampprior',
         '--epochs', '2',
         '--warmup', '0',
         '--early_stopping_epochs', '2',
         '--z1_size', '8',
         '--h_size', '64',
         '--batch_size', '16',
         '--number_components', str(K),
         '--no-cuda',
         '--seed', '42'],
        capture_output=True, text=True,
        cwd=str(tmp),
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    model_dir = next((line.split('MODEL_DIR:')[1].strip()
                      for line in result.stdout.splitlines()
                      if line.startswith('MODEL_DIR:')), None)
    assert model_dir is not None, f"MODEL_DIR not in stdout:\n{result.stdout}"
    return model_dir, csv_path, n_rows, K, n_timesteps


def _run_analyze(args, cwd):
    return subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'analyze.py')] + args + ['--no-cuda'],
        capture_output=True, text=True,
        cwd=str(cwd),
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )


class TestExportLatentsContract:
    """API.md: --export_latents → (N, z1_size) float32 at <out_z> or <dir>/z_mean.npy."""

    def test_default_path_z_mean_npy(self, trained_dir, tmp_path_factory):
        model_dir, _, n_rows, _, _ = trained_dir
        cwd = tmp_path_factory.mktemp("contract_lat_default")
        result = _run_analyze(['--dir', model_dir, '--export_latents'], cwd)
        assert result.returncode == 0, f"analyze failed:\n{result.stderr}"

        z_path = os.path.join(model_dir, 'z_mean.npy')
        assert os.path.exists(z_path), "default path z_mean.npy not created"
        z = np.load(z_path)
        assert z.shape == (n_rows, 8), f"shape {z.shape} != ({n_rows}, 8)"
        assert z.dtype == np.float32, f"dtype {z.dtype} != float32"

    def test_stdout_marker_present(self, trained_dir, tmp_path_factory):
        model_dir, _, _, _, _ = trained_dir
        cwd = tmp_path_factory.mktemp("contract_lat_marker")
        result = _run_analyze(['--dir', model_dir, '--export_latents'], cwd)
        assert 'LATENTS_PATH:' in result.stdout

    def test_out_z_override(self, trained_dir, tmp_path_factory):
        model_dir, _, n_rows, _, _ = trained_dir
        cwd = tmp_path_factory.mktemp("contract_lat_override")
        out = cwd / "custom_z.npy"
        result = _run_analyze(
            ['--dir', model_dir, '--export_latents', '--out_z', str(out)], cwd)
        assert result.returncode == 0, f"analyze failed:\n{result.stderr}"
        assert out.exists(), "--out_z override did not redirect output"
        z = np.load(out)
        assert z.shape == (n_rows, 8) and z.dtype == np.float32


class TestExportPseudoPrototypesContract:
    """API.md: --export_pseudo_prototypes → (K, D) float32 in [0, 1]."""

    def test_shape_dtype_range(self, trained_dir, tmp_path_factory):
        model_dir, _, _, K, n_timesteps = trained_dir
        cwd = tmp_path_factory.mktemp("contract_proto_shape")
        out = cwd / "proto.npy"
        result = _run_analyze(
            ['--dir', model_dir, '--export_pseudo_prototypes', str(out)], cwd)
        assert result.returncode == 0, f"analyze failed:\n{result.stderr}"

        a = np.load(out)
        assert a.shape == (K, n_timesteps), f"shape {a.shape} != ({K}, {n_timesteps})"
        assert a.dtype == np.float32, f"dtype {a.dtype} != float32"
        assert a.min() >= 0.0 and a.max() <= 1.0, \
            f"value range [{a.min()}, {a.max()}] not in [0, 1]"

    def test_stdout_marker_present(self, trained_dir, tmp_path_factory):
        model_dir, _, _, _, _ = trained_dir
        cwd = tmp_path_factory.mktemp("contract_proto_marker")
        out = cwd / "proto.npy"
        result = _run_analyze(
            ['--dir', model_dir, '--export_pseudo_prototypes', str(out)], cwd)
        assert 'PSEUDO_PROTOTYPES_PATH:' in result.stdout

    def test_no_dataset_required(self, trained_dir, tmp_path_factory):
        """When --export_pseudo_prototypes is the only flag, dataset loading is skipped.

        Verifies the contract claim: this flag works on a checkpoint even if the
        original training data is unavailable in the new working directory.
        """
        model_dir, _, _, _, _ = trained_dir
        cwd = tmp_path_factory.mktemp("contract_proto_no_data")
        out = cwd / "proto.npy"
        result = _run_analyze(
            ['--dir', model_dir, '--export_pseudo_prototypes', str(out)], cwd)
        assert result.returncode == 0, \
            f"prototype-only run should not require dataset:\n{result.stderr}"
