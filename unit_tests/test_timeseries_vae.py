"""
Unit tests for the TimeSeriesVAE model.

Tests cover:
  - Instantiation and inheritance
  - Forward pass output shapes
  - Decoder components (level, trend, seasonal, residual)
  - Gradient flow
  - VampPrior compatibility
  - Beta and Gaussian reconstruction loss
  - Generation bounds
"""

import argparse
import pytest
import torch
import numpy as np

from models.TimeSeriesVAE import TimeSeriesVAE
from utils.distributions import log_beta


def make_args(**overrides):
    defaults = dict(
        model_name='timeseries_vae',
        prior='standard',
        input_type='continuous',
        input_size=[1, 140],       # [feat_dim, seq_len]
        hidden_size=200,
        z1_size=8,
        z2_size=8,
        activation=None,
        no_attention=False,
        same_variational_var=False,
        use_logit=False,
        number_components=50,
        pseudoinputs_mean=-0.05,
        pseudoinputs_std=0.01,
        use_training_data_init=False,
        dataset_name='ecg5000',
        training_set_size=4000,
        dynamic_binarization=False,
        lr=1e-3,
        device=torch.device('cpu'),
        cuda=False,
        use_whole_train=False,
        approximate_prior=False,
        approximate_k=10,
        continuous=True,
        lambd=1e-4,
        bottleneck=6,
        K=1,
        IW=False,
        # time series specific
        seq_len=140,
        feat_dim=1,
        trend_poly=0,
        use_residual=True,
        custom_seas=None,
        no_mask=False,
        reconstruction_dist='beta',
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_model(**kwargs):
    args = make_args(**kwargs)
    return TimeSeriesVAE(args)


class TestInstantiation:
    def test_creates_successfully(self):
        model = make_model()
        assert model is not None

    def test_has_decoder_components(self):
        model = make_model()
        assert hasattr(model, 'decoder')
        assert model.decoder.level is not None
        assert model.decoder.residual is not None

    def test_trend_disabled_by_default(self):
        model = make_model()
        assert model.decoder.trend is None

    def test_trend_enabled(self):
        model = make_model(trend_poly=2)
        assert model.decoder.trend is not None

    def test_seasonal_enabled(self):
        model = make_model(custom_seas=[[24, 1]])
        assert model.decoder.seasonal is not None


class TestForwardShapes:
    def test_forward_output_shapes(self):
        model = make_model()
        x = torch.rand(8, 140)  # (batch, feat_dim * seq_len)
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        x_mean, x_logvar, (z_q, z_q_mean, z_q_logvar) = model.forward(x)

        assert x_mean.shape == (8, 140)
        assert x_logvar.shape == (8, 140)
        assert z_q.shape == (8, 8)       # (batch, z1_size)
        assert z_q_mean.shape == (8, 8)

    def test_loss_shapes(self):
        model = make_model()
        x = torch.rand(8, 140)
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)

        assert loss.dim() == 0
        assert RE.dim() == 0
        assert KL.dim() == 0
        assert torch.isfinite(loss)


class TestGradientFlow:
    def test_gradients_flow(self):
        model = make_model()
        x = torch.rand(4, 140)
        dummy_indices = torch.zeros(4, 1, dtype=torch.long)
        loss, _, _ = model.calculate_loss((x, dummy_indices), average=True)
        loss.backward()

        # p_x_mean, p_x_logvar, decoder_logstd are created by BaseModel
        # but unused by TimeSeriesVAE (which uses its own decoder) — skip them
        unused = {'p_x_mean.', 'p_x_logvar.', 'decoder_logstd'}
        for name, param in model.named_parameters():
            if param.requires_grad and not any(name.startswith(u) for u in unused):
                assert param.grad is not None, f"No gradient for {name}"


class TestReconstructionLoss:
    def test_beta_loss_default(self):
        """Default TimeSeriesVAE should use Beta reconstruction loss."""
        model = make_model()
        assert model.reconstruction_dist == 'beta'
        x = torch.rand(4, 140).clamp(1e-3, 1 - 1e-3)
        x_mean = torch.rand(4, 140).clamp(1e-3, 1 - 1e-3)
        log_conc = torch.ones(4, 140) * 3.0
        loss = model.reconstruction_loss(x, x_mean, log_conc)
        assert loss.shape == (4,)
        assert torch.isfinite(loss).all()

    def test_gaussian_loss(self):
        """Gaussian backward-compat: reconstruction_dist='gaussian'."""
        model = make_model(reconstruction_dist='gaussian')
        assert model.reconstruction_dist == 'gaussian'
        x = torch.rand(4, 140)
        x_mean = torch.rand(4, 140)
        x_logvar = torch.zeros(4, 140)
        loss = model.reconstruction_loss(x, x_mean, x_logvar)
        assert loss.shape == (4,)
        assert torch.isfinite(loss).all()

    def test_gaussian_full_pass(self):
        """Full forward+loss pass with Gaussian distribution."""
        model = make_model(reconstruction_dist='gaussian')
        x = torch.rand(8, 140)
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)
        assert torch.isfinite(loss)


class TestLogBeta:
    def test_basic_values(self):
        """log_beta should return finite negative values for valid inputs."""
        x = torch.rand(4, 10).clamp(0.01, 0.99)
        mean = torch.rand(4, 10).clamp(0.01, 0.99)
        log_conc = torch.ones(4, 10) * 3.0  # ν ≈ 20
        result = log_beta(x, mean, log_conc, dim=1)
        assert result.shape == (4,)
        assert torch.isfinite(result).all()

    def test_boundary_clamping(self):
        """x=0 and x=1 should not produce NaN/Inf thanks to clamping."""
        x = torch.tensor([[0.0, 0.5, 1.0]])
        mean = torch.tensor([[0.5, 0.5, 0.5]])
        log_conc = torch.ones(1, 3) * 3.0
        result = log_beta(x, mean, log_conc, dim=1)
        assert torch.isfinite(result).all()

    def test_higher_concentration_tighter(self):
        """Higher concentration should give higher log-prob near the mean."""
        x = torch.tensor([[0.5]])
        mean = torch.tensor([[0.5]])
        low_conc = log_beta(x, mean, torch.tensor([[2.0]]), dim=1)
        high_conc = log_beta(x, mean, torch.tensor([[5.0]]), dim=1)
        assert high_conc > low_conc


class TestGenerationBounds:
    def test_generation_in_unit_interval(self):
        """Generated values should be in [0, 1] thanks to sigmoid."""
        model = make_model()
        z = torch.randn(10, 8)
        generated = model.generate_x_from_z(z)
        assert (generated >= 0).all()
        assert (generated <= 1).all()

    def test_generation_gaussian_in_unit_interval(self):
        """Gaussian mode also uses sigmoid — generation in [0, 1]."""
        model = make_model(reconstruction_dist='gaussian')
        z = torch.randn(10, 8)
        generated = model.generate_x_from_z(z)
        assert (generated >= 0).all()
        assert (generated <= 1).all()


class TestPriorCompatibility:
    def test_vampprior(self):
        model = make_model(prior='vampprior')
        x = torch.rand(8, 140)
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_generation_standard(self):
        model = make_model()
        z = torch.randn(5, 8)
        generated = model.generate_x_from_z(z)
        assert generated.shape == (5, 140)

    def test_generation_vampprior(self):
        model = make_model(prior='vampprior')
        generated = model.generate_x(N=5)
        assert generated.shape == (5, 140)


class TestPerTimestepVariance:
    def test_variance_shape(self):
        """x_param2 should have the same shape as x_mean (per-timestep, not scalar)."""
        model = make_model()
        x = torch.rand(8, 140)
        x_mean, x_logvar, _ = model.forward(x)
        assert x_mean.shape == x_logvar.shape == (8, 140)

    def test_variance_varies_across_timesteps(self):
        """Per-timestep variance should NOT be constant across all timesteps."""
        model = make_model()
        z = torch.randn(4, 8)
        x_mean, x_param2 = model.p_x(z)
        # at init, the linear layer weights are random, so outputs should vary
        assert not torch.all(x_param2 == x_param2[:, :1]), \
            "x_param2 is constant across timesteps — not per-timestep"

    def test_variance_head_gradient_flow(self):
        """Gradients should flow through the variance head."""
        model = make_model()
        x = torch.rand(4, 140)
        dummy_indices = torch.zeros(4, 1, dtype=torch.long)
        loss, _, _ = model.calculate_loss((x, dummy_indices), average=True)
        loss.backward()
        for name, param in model.variance_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for variance_head.{name}"

    def test_beta_concentration_clamped(self):
        """Beta mode: log-concentration should be in [0.5, 8.0]."""
        model = make_model(reconstruction_dist='beta')
        z = torch.randn(4, 8)
        _, x_param2 = model.p_x(z)
        assert (x_param2 >= 0.5).all()
        assert (x_param2 <= 8.0).all()

    def test_gaussian_logvar_clamped(self):
        """Gaussian mode: logvar should be in [-4.5, 0.0]."""
        model = make_model(reconstruction_dist='gaussian')
        z = torch.randn(4, 8)
        _, x_param2 = model.p_x(z)
        assert (x_param2 >= -4.5).all()
        assert (x_param2 <= 0.0).all()


class TestMultivariate:
    def test_multivariate_forward(self):
        """Test with multiple features per timestep."""
        model = make_model(feat_dim=3, input_size=[3, 50], seq_len=50)
        x = torch.rand(4, 150)  # 3 features * 50 timesteps
        dummy_indices = torch.zeros(4, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)
        assert torch.isfinite(loss)
