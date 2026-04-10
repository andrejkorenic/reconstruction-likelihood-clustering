"""Tests for utils/visual_recon.py -- reconstruction likelihood visualization."""
import glob as globmod
import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace


class MockVAE(nn.Module):
    """Minimal VAE mock that supports forward() -> (x_mean, x_logvar, latent_stats)."""

    def __init__(self, input_dim=784, z_dim=8, input_type='binary'):
        super().__init__()
        self.args = SimpleNamespace(
            input_size=[1, 28, 28],
            input_type=input_type,
            z1_size=z_dim,
            model_name='vae',
            K=1,
        )
        self.enc = nn.Linear(input_dim, z_dim * 2)
        self.dec = nn.Linear(z_dim, input_dim)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = self.enc(x_flat)
        z_mean, z_logvar = h[:, :self.args.z1_size], h[:, self.args.z1_size:]
        std = torch.exp(0.5 * z_logvar)
        z = z_mean + std * torch.randn_like(std)
        x_mean = torch.sigmoid(self.dec(z))
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar, (z, z_mean, z_logvar)

    def reconstruction_loss(self, x, x_mean, x_logvar):
        """Per-sample log-likelihood (sum over pixels). Higher = better."""
        x_flat = x.view(x.size(0), -1)
        bce = x_flat * torch.log(x_mean + 1e-8) + (1 - x_flat) * torch.log(1 - x_mean + 1e-8)
        return bce.sum(dim=1)


@pytest.fixture
def mock_args():
    return SimpleNamespace(
        input_size=[1, 28, 28],
        input_type='binary',
        z1_size=8,
        model_name='vae',
        K=1,
        device=torch.device('cpu'),
    )


@pytest.fixture
def mock_model():
    model = MockVAE(input_dim=784, z_dim=8, input_type='binary')
    model.eval()
    return model


@pytest.fixture
def mock_model_gray():
    model = MockVAE(input_dim=784, z_dim=8, input_type='gray')
    model.args.input_type = 'gray'
    model.eval()
    return model


def _make_test_loader(n_samples, n_classes, input_size=(1, 28, 28)):
    """Create a DataLoader with n_samples split evenly across n_classes."""
    data = torch.rand(n_samples, *input_size)
    if n_classes <= 1:
        labels = torch.zeros(n_samples, dtype=torch.long)
    else:
        labels = torch.arange(n_classes).repeat(n_samples // n_classes)
        if len(labels) < n_samples:
            labels = torch.cat([labels, torch.zeros(n_samples - len(labels), dtype=torch.long)])
    dataset = torch.utils.data.TensorDataset(data, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
from utils.visual_recon import plot_reconstruction_likelihood


class TestOutputStructure:
    def test_creates_per_class_files(self, mock_model, mock_args, tmp_path):
        """Two classes -> class_0_best.png, class_0_worst.png, class_1_best.png, class_1_worst.png."""
        test_loader = _make_test_loader(n_samples=20, n_classes=2)
        output_dir = str(tmp_path) + '/'

        plot_reconstruction_likelihood(mock_args, mock_model, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert recon_dir.exists()
        assert (recon_dir / 'class_0_best.png').exists()
        assert (recon_dir / 'class_0_worst.png').exists()
        assert (recon_dir / 'class_1_best.png').exists()
        assert (recon_dir / 'class_1_worst.png').exists()
        assert not (recon_dir / 'overall_best.png').exists()

    def test_no_class_fallback(self, mock_model, mock_args, tmp_path):
        """All labels=0 → overall_best.png and overall_worst.png, no class_* files."""
        test_loader = _make_test_loader(n_samples=20, n_classes=1)
        output_dir = str(tmp_path) + '/'

        plot_reconstruction_likelihood(mock_args, mock_model, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert (recon_dir / 'overall_best.png').exists()
        assert (recon_dir / 'overall_worst.png').exists()
        assert not (recon_dir / 'class_0_best.png').exists()

    def test_fewer_than_5_samples(self, mock_model, mock_args, tmp_path):
        """Class with only 3 samples → file created, no crash."""
        data = torch.rand(6, 1, 28, 28)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        dataset = torch.utils.data.TensorDataset(data, labels)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=6)

        output_dir = str(tmp_path) + '/'
        plot_reconstruction_likelihood(mock_args, mock_model, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert (recon_dir / 'class_0_best.png').exists()
        assert (recon_dir / 'class_0_worst.png').exists()
        assert (recon_dir / 'class_1_best.png').exists()
        assert (recon_dir / 'class_1_worst.png').exists()


class TestIWAEGuard:
    def test_K_restored_after_call(self, mock_model, mock_args, tmp_path):
        """args.K must be restored to original value after function returns."""
        mock_args.K = 5
        test_loader = _make_test_loader(n_samples=10, n_classes=2)
        output_dir = str(tmp_path) + '/'

        plot_reconstruction_likelihood(mock_args, mock_model, test_loader, output_dir)

        assert mock_args.K == 5, f"args.K was {mock_args.K}, expected 5"

    def test_K_restored_even_on_error(self, mock_args, tmp_path):
        """args.K must be restored even if the model raises an error."""
        class BrokenModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.args = SimpleNamespace(K=1)

            def forward(self, x):
                raise RuntimeError("intentional failure")

        mock_args.K = 7
        test_loader = _make_test_loader(n_samples=10, n_classes=2)
        output_dir = str(tmp_path) + '/'

        with pytest.raises(RuntimeError, match="intentional failure"):
            plot_reconstruction_likelihood(mock_args, BrokenModel(), test_loader, output_dir)

        assert mock_args.K == 7, f"args.K was {mock_args.K}, expected 7"


class TestInputTypes:
    def test_binary_produces_files(self, mock_model, mock_args, tmp_path):
        """Binary input type completes and produces output files."""
        mock_args.input_type = 'binary'
        test_loader = _make_test_loader(n_samples=10, n_classes=2)
        output_dir = str(tmp_path) + '/'

        plot_reconstruction_likelihood(mock_args, mock_model, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert (recon_dir / 'class_0_best.png').exists()

    def test_gray_produces_files(self, mock_model_gray, tmp_path):
        """Gray input type completes and produces output files."""
        args = SimpleNamespace(
            input_size=[1, 28, 28],
            input_type='gray',
            z1_size=8,
            model_name='vae',
            K=1,
            device=torch.device('cpu'),
        )
        test_loader = _make_test_loader(n_samples=10, n_classes=2)
        output_dir = str(tmp_path) + '/'

        plot_reconstruction_likelihood(args, mock_model_gray, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert (recon_dir / 'class_0_best.png').exists()

    def test_log_likelihoods_are_finite(self, mock_model, mock_args, tmp_path):
        """Log-likelihoods computed during visualization must be finite (no NaN/Inf)."""
        test_loader = _make_test_loader(n_samples=10, n_classes=2)

        from utils.visual_recon import _encode_decode_all
        _, _, _, log_likelihoods = _encode_decode_all(mock_args, mock_model, test_loader)

        assert np.all(np.isfinite(log_likelihoods)), \
            f"Non-finite log-likelihoods found: {log_likelihoods}"


# ---------------------------------------------------------------------------
# Time series mock & tests
# ---------------------------------------------------------------------------

class MockTimeSeriesVAE(nn.Module):
    """Minimal TimeSeriesVAE mock. input_size=[feat_dim, seq_len]."""

    def __init__(self, feat_dim=1, seq_len=50, z_dim=8):
        super().__init__()
        self.args = SimpleNamespace(
            input_size=[feat_dim, seq_len],
            input_type='continuous',
            z1_size=z_dim,
            model_name='timeseries_vae',
            K=1,
        )
        D = feat_dim * seq_len
        self.enc = nn.Linear(D, z_dim * 2)
        self.dec = nn.Linear(z_dim, D)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        h = self.enc(x_flat)
        z_mean, z_logvar = h[:, :self.args.z1_size], h[:, self.args.z1_size:]
        std = torch.exp(0.5 * z_logvar)
        z = z_mean + std * torch.randn_like(std)
        x_mean = torch.sigmoid(self.dec(z))
        x_logvar = torch.zeros_like(x_mean)
        return x_mean, x_logvar, (z, z_mean, z_logvar)

    def reconstruction_loss(self, x, x_mean, x_logvar):
        x_flat = x.view(x.size(0), -1)
        diff = x_flat - x_mean
        return -(diff ** 2).sum(dim=1)


@pytest.fixture
def ts_args():
    return SimpleNamespace(
        input_size=[1, 50],
        input_type='continuous',
        z1_size=8,
        model_name='timeseries_vae',
        K=1,
        plot_timesteps=None,
        device=torch.device('cpu'),
    )


@pytest.fixture
def ts_model():
    model = MockTimeSeriesVAE(feat_dim=1, seq_len=50, z_dim=8)
    model.eval()
    return model


class TestTimeSeriesReconViz:
    def test_ts_per_class_creates_pngs(self, ts_model, ts_args, tmp_path):
        """2 classes, 20 samples -> class_0/1 best/worst PNGs, no CSV."""
        test_loader = _make_test_loader(n_samples=20, n_classes=2,
                                        input_size=(1, 50))
        output_dir = str(tmp_path) + '/'

        plot_reconstruction_likelihood(ts_args, ts_model, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert (recon_dir / 'class_0_best.png').exists()
        assert (recon_dir / 'class_0_worst.png').exists()
        assert (recon_dir / 'class_1_best.png').exists()
        assert (recon_dir / 'class_1_worst.png').exists()
        # No CSV files should be produced
        csv_files = list(recon_dir.glob('*.csv'))
        assert len(csv_files) == 0, f"Unexpected CSV files: {csv_files}"

    def test_ts_no_class_fallback(self, ts_model, ts_args, tmp_path):
        """All labels=0 -> overall_best/worst PNGs, no class_0_* files."""
        test_loader = _make_test_loader(n_samples=20, n_classes=1,
                                        input_size=(1, 50))
        output_dir = str(tmp_path) + '/'

        plot_reconstruction_likelihood(ts_args, ts_model, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert (recon_dir / 'overall_best.png').exists()
        assert (recon_dir / 'overall_worst.png').exists()
        assert not (recon_dir / 'class_0_best.png').exists()

    def test_ts_no_csv_produced(self, ts_model, ts_args, tmp_path):
        """Regression guard: no reconstruction_likelihoods.csv from old behavior."""
        test_loader = _make_test_loader(n_samples=10, n_classes=2,
                                        input_size=(1, 50))
        output_dir = str(tmp_path) + '/'

        plot_reconstruction_likelihood(ts_args, ts_model, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert not (recon_dir / 'reconstruction_likelihoods.csv').exists()

    def test_ts_fewer_than_5_samples(self, ts_model, ts_args, tmp_path):
        """3 samples per class, 2 classes -> no crash, PNGs created."""
        data = torch.rand(6, 1, 50)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        dataset = torch.utils.data.TensorDataset(data, labels)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=6)

        output_dir = str(tmp_path) + '/'
        plot_reconstruction_likelihood(ts_args, ts_model, test_loader, output_dir)

        recon_dir = tmp_path / 'recon_likelihood'
        assert (recon_dir / 'class_0_best.png').exists()
        assert (recon_dir / 'class_0_worst.png').exists()
        assert (recon_dir / 'class_1_best.png').exists()
        assert (recon_dir / 'class_1_worst.png').exists()

    def test_ts_log_likelihoods_finite(self, ts_model, ts_args):
        """Log-likelihoods from _encode_decode_all must be finite for TS data."""
        test_loader = _make_test_loader(n_samples=10, n_classes=2,
                                        input_size=(1, 50))

        from utils.visual_recon import _encode_decode_all
        _, _, _, log_likelihoods = _encode_decode_all(ts_args, ts_model,
                                                       test_loader)

        assert np.all(np.isfinite(log_likelihoods)), \
            f"Non-finite log-likelihoods found: {log_likelihoods}"
