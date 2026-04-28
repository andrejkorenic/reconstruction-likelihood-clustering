"""
Tests for utils/ood_scores.py — per-level ELBO decomposition for OOD detection.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.HVAE_2level import VAE as HVAE_2level

REPO_ROOT = str(Path(__file__).resolve().parents[1])


# ======================================================================================================================
# Helper: minimal args Namespace for HVAE
# ======================================================================================================================
def make_args(**overrides):
    defaults = dict(
        model_name='hvae_2level',
        prior='standard',
        input_type='binary',
        input_size=[1, 28, 28],
        hidden_size=300,
        z1_size=40,
        z2_size=40,
        activation=None,
        no_attention=False,
        same_variational_var=False,
        use_logit=False,
        number_components=500,
        pseudoinputs_mean=-0.05,
        pseudoinputs_std=0.01,
        use_training_data_init=False,
        dataset_name='dynamic_mnist',
        training_set_size=60000,
        dynamic_binarization=False,
        lr=5e-4,
        device=torch.device('cpu'),
        cuda=False,
        use_whole_train=False,
        approximate_prior=False,
        approximate_k=10,
        continuous=False,
        lambd=1e-4,
        bottleneck=6,
        K=1,
        IW=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_hvae(**kwargs):
    args = make_args(**kwargs)
    model = HVAE_2level(args)
    model.eval()
    return model


def make_test_loader(n=32, input_dim=784, batch_size=16):
    """Create a test-format DataLoader: yields (data, target) — no indices.

    Uses a dedicated Generator so DataLoader internals don't consume
    the global torch RNG (important for reproducible consistency tests).
    """
    data = torch.rand(n, input_dim)
    targets = torch.zeros(n, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(data, targets)
    gen = torch.Generator()
    gen.manual_seed(0)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, generator=gen)


# ======================================================================================================================
# decompose_elbo: return structure
# ======================================================================================================================
class TestDecomposeElboStructure:
    def test_returns_dict_with_correct_keys(self):
        from utils.ood_scores import decompose_elbo
        model = make_hvae()
        loader = make_test_loader(n=16)
        result = decompose_elbo(model, loader, torch.device('cpu'))
        assert set(result.keys()) == {'re', 'kl1', 'kl2'}

    def test_returns_correct_number_of_samples(self):
        from utils.ood_scores import decompose_elbo
        model = make_hvae()
        n = 24
        loader = make_test_loader(n=n, batch_size=8)
        result = decompose_elbo(model, loader, torch.device('cpu'))
        assert len(result['re']) == n
        assert len(result['kl1']) == n
        assert len(result['kl2']) == n

    def test_values_are_finite(self):
        from utils.ood_scores import decompose_elbo
        model = make_hvae()
        loader = make_test_loader(n=16)
        result = decompose_elbo(model, loader, torch.device('cpu'))
        for key in ('re', 'kl1', 'kl2'):
            assert all(np.isfinite(v) for v in result[key]), f'{key} has non-finite values'


# ======================================================================================================================
# decompose_elbo: numerical consistency with calculate_loss
# ======================================================================================================================
class TestDecomposeElboConsistency:
    def test_kl_sum_matches_calculate_loss(self):
        """KL1 + KL2 from decompose_elbo should match KL from calculate_loss.

        Both paths get the same data and the same torch seed before
        model.forward(), so the reparameterized z samples are identical.
        """
        from utils.ood_scores import decompose_elbo
        torch.manual_seed(42)
        model = make_hvae()

        # Build a fixed-data loader with isolated generator so DataLoader
        # internals don't touch the global RNG.
        data = torch.rand(8, 784)
        targets = torch.zeros(8, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(data, targets)
        dl_gen = torch.Generator().manual_seed(0)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, generator=dl_gen)

        # Path 1: decompose_elbo — seed before so reparameterize draws are deterministic
        torch.manual_seed(0)
        result = decompose_elbo(model, loader, torch.device('cpu'))
        kl_decomposed = [result['kl1'][i] + result['kl2'][i] for i in range(len(result['kl1']))]

        # Path 2: calculate_loss — same seed yields same z samples
        torch.manual_seed(0)
        model.eval()
        with torch.no_grad():
            dummy_indices = torch.zeros(data.size(0), 1, dtype=torch.long)
            _, RE_calc, KL_calc = model.calculate_loss(
                (data, dummy_indices), beta=1., average=False)

        kl_from_loss = KL_calc.tolist()
        for d, l in zip(kl_decomposed, kl_from_loss):
            assert abs(d - l) < 1e-4, f'KL mismatch: decomposed={d:.6f}, calculate_loss={l:.6f}'

    def test_re_matches_calculate_loss(self):
        """RE from decompose_elbo should match RE from calculate_loss.

        Both return log p(x|z) (negative). calculate_loss stores it as-is
        in the RE return value; decompose_elbo returns it as 're'.
        """
        from utils.ood_scores import decompose_elbo
        torch.manual_seed(42)
        model = make_hvae()

        # Build a fixed-data loader with isolated generator
        data = torch.rand(8, 784)
        targets = torch.zeros(8, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(data, targets)
        dl_gen = torch.Generator().manual_seed(0)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, generator=dl_gen)

        # Path 1: decompose_elbo — seed before for deterministic reparameterize
        torch.manual_seed(0)
        result = decompose_elbo(model, loader, torch.device('cpu'))

        # Path 2: calculate_loss — same seed yields same z samples
        torch.manual_seed(0)
        model.eval()
        with torch.no_grad():
            dummy_indices = torch.zeros(data.size(0), 1, dtype=torch.long)
            _, RE_calc, _ = model.calculate_loss(
                (data, dummy_indices), beta=1., average=False)

        # Both are log p(x|z) — same sign, should match directly
        re_from_loss = RE_calc.tolist()
        re_decomposed = result['re']
        for d, l in zip(re_decomposed, re_from_loss):
            assert abs(d - l) < 1e-4, f'RE mismatch: decomposed={d:.6f}, calculate_loss={l:.6f}'

    def test_kl1_is_nonnegative(self):
        """KL1 should be non-negative on average (single-sample MC may fluctuate)."""
        from utils.ood_scores import decompose_elbo
        model = make_hvae()
        loader = make_test_loader(n=64)
        result = decompose_elbo(model, loader, torch.device('cpu'))
        mean_kl1 = np.mean(result['kl1'])
        assert mean_kl1 >= 0, f'Mean KL1 is negative: {mean_kl1}'

    def test_kl2_is_nonnegative(self):
        """KL2 should be non-negative on average (single-sample MC may fluctuate)."""
        from utils.ood_scores import decompose_elbo
        model = make_hvae()
        loader = make_test_loader(n=64)
        result = decompose_elbo(model, loader, torch.device('cpu'))
        mean_kl2 = np.mean(result['kl2'])
        assert mean_kl2 >= 0, f'Mean KL2 is negative: {mean_kl2}'


# ======================================================================================================================
# save_ood_scores: CSV output
# ======================================================================================================================
class TestSaveOodScores:
    def _make_scores(self, n=10):
        return {
            're': [-100.0 + i for i in range(n)],
            'kl1': [10.0 + i * 0.5 for i in range(n)],
            'kl2': [5.0 + i * 0.3 for i in range(n)],
        }

    def test_csv_has_correct_header(self, tmp_path):
        from utils.ood_scores import save_ood_scores
        path = str(tmp_path / 'scores.csv')
        save_ood_scores(self._make_scores(), None, path)
        with open(path) as f:
            header = f.readline().strip()
        assert header == 'dataset,sample_idx,re,kl1,kl2,elbo,l_above_1'

    def test_csv_id_only_row_count(self, tmp_path):
        from utils.ood_scores import save_ood_scores
        n = 10
        path = str(tmp_path / 'scores.csv')
        save_ood_scores(self._make_scores(n), None, path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == n + 1  # header + n data rows

    def test_csv_with_ood_has_both_datasets(self, tmp_path):
        from utils.ood_scores import save_ood_scores
        path = str(tmp_path / 'scores.csv')
        save_ood_scores(self._make_scores(5), self._make_scores(8), path)
        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 5 + 8 + 1  # header + id + ood
        datasets = [row[0] for row in rows[1:]]
        assert datasets[:5] == ['id'] * 5
        assert datasets[5:] == ['ood'] * 8

    def test_csv_elbo_is_re_minus_kl1_minus_kl2(self, tmp_path):
        from utils.ood_scores import save_ood_scores
        scores = self._make_scores(3)
        path = str(tmp_path / 'scores.csv')
        save_ood_scores(scores, None, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                expected_elbo = scores['re'][i] - scores['kl1'][i] - scores['kl2'][i]
                assert abs(float(row['elbo']) - expected_elbo) < 1e-3

    def test_csv_l_above_1_is_re_minus_kl2(self, tmp_path):
        from utils.ood_scores import save_ood_scores
        scores = self._make_scores(3)
        path = str(tmp_path / 'scores.csv')
        save_ood_scores(scores, None, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                expected = scores['re'][i] - scores['kl2'][i]
                assert abs(float(row['l_above_1']) - expected) < 1e-3


# ======================================================================================================================
# plot_ood_histograms: file creation
# ======================================================================================================================
class TestPlotOodHistograms:
    def _make_scores(self, n=100):
        return {
            're': np.random.randn(n).tolist(),
            'kl1': np.abs(np.random.randn(n)).tolist(),
            'kl2': np.abs(np.random.randn(n)).tolist(),
        }

    def test_creates_png_file(self, tmp_path):
        from utils.ood_scores import plot_ood_histograms
        path = str(tmp_path / 'hist.png')
        plot_ood_histograms(self._make_scores(), self._make_scores(), path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


# ======================================================================================================================
# CLI integration: --ood_scores flag in analyze.py
# ======================================================================================================================
class TestCLIIntegration:
    def test_ood_scores_flag_is_accepted(self):
        """analyze.py should accept --ood_scores without crashing on arg parsing."""
        import subprocess
        result = subprocess.run(
            [sys.executable, 'analyze.py', '--help'],
            capture_output=True, text=True, timeout=10,
            cwd=REPO_ROOT
        )
        assert result.returncode == 0
        assert '--ood_scores' in result.stdout

    def test_ood_dataset_flag_is_accepted(self):
        """analyze.py should accept --ood_dataset."""
        import subprocess
        result = subprocess.run(
            [sys.executable, 'analyze.py', '--help'],
            capture_output=True, text=True, timeout=10,
            cwd=REPO_ROOT
        )
        assert result.returncode == 0
        assert '--ood_dataset' in result.stdout
