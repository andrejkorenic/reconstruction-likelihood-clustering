"""End-to-end smoke tests for all model architectures and analysis pipelines.

Runs real training (5 epochs, tiny settings) via subprocess for every
model/dataset/prior combination, then verifies that analyze.py and LL
estimation work on the resulting checkpoints.

Usage:
    # Collect tests without running (should show 16 deselected):
    uv run pytest unit_tests/test_smoke_all_models.py --collect-only

    # Run all smoke tests (~15-30 min depending on hardware):
    uv run pytest unit_tests/test_smoke_all_models.py --run-slow -x -v
"""
# Known bugs found by this test suite (fix these, then re-enable tests):
# - BUG-A: FIXED — model.forward() used universally in visual_recon.py
# - BUG-B: FIXED — ConvVAE q_z auto-reshapes flat input; TimeSeriesVAE p_x() flattens extra batch dims.
# - BUG-C: FIXED — run.py --dir now loads config before dataset.
# - BUG-D: FIXED — analyze.py now maps classifier args to classify_data.py expected names

import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared lightweight settings for all training runs
# ---------------------------------------------------------------------------
# NOTE: --MB is density_estimation.py only; run.py does not accept it.
BASE_ARGS = [
    '--epochs', '5', '--warmup', '3', '--batch_size', '50',
    '--z1_size', '8', '--z2_size', '8', '--h_size', '32',
    '--number_components', '10', '--S', '10',
    '--early_stopping_epochs', '5', '--no_ll',
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _find_latest_snap_dir(model_name):
    """Find the most recently created checkpoint directory for a model.

    Directory convention (see utils/create_model.py):
        pretrained_models/{dataset}_{prior}_model_name={model_name}/{timestamp}/
    """
    pattern = f'pretrained_models/*_model_name={model_name}/*/'
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def _train(extra_args, timeout=180):
    """Train a model via run.py subprocess."""
    result = subprocess.run(
        [sys.executable, 'run.py'] + BASE_ARGS + extra_args,
        capture_output=True, text=True, timeout=timeout)
    return result


def _analyze(snap_dir, flags, timeout=120):
    """Run analyze.py on a checkpoint dir with given flags."""
    result = subprocess.run(
        [sys.executable, 'analyze.py', '--dir', snap_dir] + flags,
        capture_output=True, text=True, timeout=timeout)
    return result


def _ll(snap_dir, timeout=120):
    """Run LL estimation via run.py --dir --ll."""
    result = subprocess.run(
        [sys.executable, 'run.py', '--dir', snap_dir, '--ll', '--S', '10'],
        capture_output=True, text=True, timeout=timeout)
    return result


def _assert_train_ok(result, model_name):
    """Assert training succeeded and return snap_dir."""
    assert result.returncode == 0, (
        f"Training {model_name} failed (code {result.returncode}).\n"
        f"STDERR: {result.stderr[-500:]}\nSTDOUT: {result.stdout[-500:]}")
    snap_dir = _find_latest_snap_dir(model_name)
    assert snap_dir is not None, f"No checkpoint found for {model_name}"
    return snap_dir


# ===================================================================
# Core pipeline tests (10) -- train + analyze + ll
# ===================================================================

@pytest.mark.slow
def test_vae_standard():
    result = _train(['--model_name', 'vae', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'standard', '--dynamic_binarization'])
    snap_dir = _assert_train_ok(result, 'vae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()
    assert any(p.glob('recon_likelihood/*.png'))
    assert any(p.glob('generated/*.png'))


@pytest.mark.slow
def test_vae_vampprior():
    result = _train(['--model_name', 'vae', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'vampprior', '--dynamic_binarization'])
    snap_dir = _assert_train_ok(result, 'vae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


@pytest.mark.slow
def test_vae_exemplar():
    result = _train(['--model_name', 'vae', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'exemplar_prior', '--dynamic_binarization',
                     '--approximate_prior', '--approximate_k', '5'])
    snap_dir = _assert_train_ok(result, 'vae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


@pytest.mark.slow
def test_convvae_standard():
    result = _train(['--model_name', 'convvae', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'standard', '--dynamic_binarization'])
    snap_dir = _assert_train_ok(result, 'convvae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


@pytest.mark.slow
def test_iwae_standard():
    result = _train(['--model_name', 'iwae', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'standard', '--dynamic_binarization', '--K', '5'])
    snap_dir = _assert_train_ok(result, 'iwae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


@pytest.mark.slow
def test_iwae_2level_standard():
    result = _train(['--model_name', 'iwae_2level', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'standard', '--dynamic_binarization'])
    snap_dir = _assert_train_ok(result, 'iwae_2level')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


@pytest.mark.slow
def test_hvae_2level_standard():
    result = _train(['--model_name', 'hvae_2level', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'standard', '--dynamic_binarization'])
    snap_dir = _assert_train_ok(result, 'hvae_2level')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


@pytest.mark.slow
def test_convhvae_2level_standard():
    result = _train(['--model_name', 'convhvae_2level', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'standard', '--dynamic_binarization'])
    snap_dir = _assert_train_ok(result, 'convhvae_2level')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


@pytest.mark.slow
def test_timeseries_vae_standard():
    result = _train(['--model_name', 'timeseries_vae', '--dataset_name', 'ecg5000',
                     '--prior', 'standard', '--reconstruction_dist', 'beta'])
    snap_dir = _assert_train_ok(result, 'timeseries_vae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()
    assert (p / 'recon_likelihood' / 'reconstruction_likelihoods.csv').exists()


@pytest.mark.slow
def test_timeseries_vae_vampprior():
    result = _train(['--model_name', 'timeseries_vae', '--dataset_name', 'ecg5000',
                     '--prior', 'vampprior', '--reconstruction_dist', 'beta'])
    snap_dir = _assert_train_ok(result, 'timeseries_vae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


# ===================================================================
# Cross-dataset tests (2)
# ===================================================================

@pytest.mark.slow
def test_vae_fashion_mnist():
    result = _train(['--model_name', 'vae', '--dataset_name', 'fashion_mnist',
                     '--prior', 'standard'])
    snap_dir = _assert_train_ok(result, 'vae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


@pytest.mark.slow
def test_vae_cifar10():
    # cifar10 is continuous RGB 3x32x32 -- no --dynamic_binarization
    result = _train(['--model_name', 'vae', '--dataset_name', 'cifar10',
                     '--prior', 'standard', '--h_size', '128'])
    snap_dir = _assert_train_ok(result, 'vae')

    # analyze.py should work (flat VAE handles any input_size)
    result = _analyze(snap_dir, ['--cluster', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    result = _ll(snap_dir)
    assert result.returncode == 0, f"LL estimation failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert (p / 'll_metrics.csv').exists()


# ===================================================================
# Special flag tests (4)
# ===================================================================

@pytest.mark.slow
def test_ood_scores_hvae():
    """Train hvae_2level, then run OOD score decomposition."""
    result = _train(['--model_name', 'hvae_2level', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'standard', '--dynamic_binarization'])
    snap_dir = _assert_train_ok(result, 'hvae_2level')

    result = _analyze(snap_dir, ['--ood_scores', '--ood_dataset', 'fashion_mnist'])
    assert result.returncode == 0, f"OOD analysis failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'ood_scores.csv').exists()
    assert (p / 'ood_histograms.png').exists()


@pytest.mark.slow
def test_knn_classify():
    """Train vae standard, then run KNN and classifier on latent space."""
    result = _train(['--model_name', 'vae', '--dataset_name', 'dynamic_mnist',
                     '--prior', 'standard', '--dynamic_binarization'])
    snap_dir = _assert_train_ok(result, 'vae')

    result = _analyze(snap_dir, ['--KNN', '--classify', '--classify_epochs', '2'])
    assert result.returncode == 0, f"KNN/classify failed:\n{result.stderr[-500:]}"


@pytest.mark.slow
def test_run_chaining():
    """run.py with --cluster and --generate flags (post-training dispatch)."""
    # These flags are handled by run.py -> perform_experiment -> final_evaluation
    extra = ['--model_name', 'vae', '--dataset_name', 'dynamic_mnist',
             '--prior', 'standard', '--dynamic_binarization',
             '--cluster', '--generate']
    result = _train(extra, timeout=240)
    snap_dir = _assert_train_ok(result, 'vae')

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
    assert any(p.glob('generated/*.png'))


@pytest.mark.slow
def test_timeseries_gaussian():
    """Train timeseries_vae with gaussian reconstruction, then analyze."""
    result = _train(['--model_name', 'timeseries_vae', '--dataset_name', 'ecg5000',
                     '--prior', 'standard', '--reconstruction_dist', 'gaussian'])
    snap_dir = _assert_train_ok(result, 'timeseries_vae')

    result = _analyze(snap_dir, ['--cluster', '--recon_viz', '--generate'])
    assert result.returncode == 0, f"Analysis failed:\n{result.stderr[-500:]}"

    p = Path(snap_dir)
    assert (p / 'cluster_metrics.csv').exists()
