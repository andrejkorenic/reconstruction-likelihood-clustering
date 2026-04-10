"""
Unit tests for reparameterization and K-sample resampling.

Tests cover:
  - reparameterize() output shapes for K=1 and K>1
  - gradient flow through reparameterization
  - K-sample loss aggregation in calculate_loss()
  - NotImplementedError guards for unsupported combinations
"""

import argparse

import pytest
import torch
import numpy as np

from models.AbsModel import AbsModel
from models.AbsHModel import BaseHModel

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
# Helper: minimal args Namespace
# ======================================================================================================================
def make_args(**overrides):
    defaults = dict(
        model_name='vae',
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


def make_vae(K=1, **kwargs):
    """Instantiate a VAE with the given K value."""
    from models.VAE import VAE
    args = make_args(K=K, **kwargs)
    model = VAE(args)
    return model


def make_hvae(K=1, **kwargs):
    """Instantiate a hierarchical VAE with the given K value."""
    from models.HVAE_2level import VAE
    args = make_args(model_name='hvae_2level', K=K, **kwargs)
    model = VAE(args)
    return model


# ======================================================================================================================
# reparameterize() — shapes and gradients
# ======================================================================================================================
class TestReparameterize:
    def test_k1_shape(self):
        model = make_vae(K=1)
        mu = torch.randn(8, 40)
        logvar = torch.randn(8, 40)
        z = model.reparameterize(mu, logvar)
        assert z.shape == (8, 40)

    def test_k5_shape(self):
        model = make_vae(K=5)
        model.resample = True  # IWAE would set this
        mu = torch.randn(8, 40)
        logvar = torch.randn(8, 40)
        z = model.reparameterize(mu, logvar)
        assert z.shape == (5 * 8, 40)

    def test_k1_gradient_flows(self):
        model = make_vae(K=1)
        mu = torch.randn(4, 40, requires_grad=True)
        logvar = torch.randn(4, 40, requires_grad=True)
        z = model.reparameterize(mu, logvar)
        loss = z.sum()
        loss.backward()
        assert mu.grad is not None
        assert logvar.grad is not None

    def test_k5_gradient_flows(self):
        model = make_vae(K=5)
        model.resample = True  # IWAE would set this
        mu = torch.randn(4, 40, requires_grad=True)
        logvar = torch.randn(4, 40, requires_grad=True)
        z = model.reparameterize(mu, logvar)
        loss = z.sum()
        loss.backward()
        assert mu.grad is not None
        assert logvar.grad is not None

    def test_resample_flag(self):
        """resample is always False for non-IWAE models, regardless of K."""
        model_k1 = make_vae(K=1)
        model_k5 = make_vae(K=5)
        assert model_k1.resample is False
        assert model_k5.resample is False

    def test_default_k_is_1(self):
        """Model created without K arg defaults to K=1, resample=False."""
        args = make_args()
        del args.K  # simulate missing K (e.g. old code)
        from models.VAE import VAE
        model = VAE(args)
        assert model.resample is False


# ======================================================================================================================
# forward() — K-sample expansion of latent stats
# ======================================================================================================================
class TestForwardKSample:
    def test_k1_forward_shapes(self):
        model = make_vae(K=1)
        x = torch.rand(8, np.prod(model.args.input_size))
        x_mean, x_logvar, (z_q, z_q_mean, z_q_logvar) = model.forward(x)
        assert z_q.shape == (8, 40)
        assert z_q_mean.shape == (8, 40)
        assert z_q_logvar.shape == (8, 40)
        assert x_mean.shape == (8, np.prod(model.args.input_size))

    def test_k5_forward_shapes(self):
        model = make_vae(K=5)
        model.resample = True  # IWAE would set this
        x = torch.rand(8, np.prod(model.args.input_size))
        x_mean, x_logvar, (z_q, z_q_mean, z_q_logvar) = model.forward(x)
        # z_q, z_q_mean, z_q_logvar are all (K*batch, z_dim)
        assert z_q.shape == (5 * 8, 40)
        assert z_q_mean.shape == (5 * 8, 40)
        assert z_q_logvar.shape == (5 * 8, 40)
        # x_mean is (K*batch, D) because p_x receives (K*batch, z_dim) input
        assert x_mean.shape == (5 * 8, np.prod(model.args.input_size))


# ======================================================================================================================
# calculate_loss() — K-sample aggregation
# ======================================================================================================================
class TestCalculateLossKSample:
    def test_k1_loss_shape_no_average(self):
        model = make_vae(K=1)
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=False)
        assert loss.shape == (8,)

    def test_k5_loss_shape_no_average(self):
        model = make_vae(K=5)
        model.resample = True  # IWAE would set this
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=False)
        # after K-averaging, shape should be (batch,)
        assert loss.shape == (8,)
        assert RE.shape == (8,)
        assert KL.shape == (8,)

    def test_k5_loss_scalar_with_average(self):
        model = make_vae(K=5)
        model.resample = True  # IWAE would set this
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)
        assert loss.dim() == 0
        assert RE.dim() == 0
        assert KL.dim() == 0

    def test_k5_vampprior_works(self):
        model = make_vae(K=5, prior='vampprior')
        model.resample = True  # IWAE would set this
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)
        assert loss.dim() == 0


# ======================================================================================================================
# Guards — NotImplementedError for unsupported combinations
# ======================================================================================================================
class TestGuards:
    def test_k_gt1_hierarchical_raises(self):
        model = make_hvae(K=5)
        model.resample = True  # IWAE would set this
        x = torch.rand(8, np.prod(model.args.input_size))
        with pytest.raises(NotImplementedError, match="hierarchical"):
            model.forward(x)

    def test_k1_hierarchical_works(self):
        model = make_hvae(K=1)
        x = torch.rand(8, np.prod(model.args.input_size))
        # should not raise
        x_mean, x_logvar, latent_stats = model.forward(x)
        assert x_mean is not None

    def test_k_gt1_exemplar_prior_raises(self):
        model = make_vae(K=5, prior='exemplar_prior')
        model.resample = True  # IWAE would set this
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        with pytest.raises(NotImplementedError, match="exemplar_prior"):
            model.calculate_loss((x, dummy_indices))

    def test_iw_flag_on_vae_raises(self):
        """--IW on a non-IWAE model should raise ValueError directing to --model_name iwae."""
        model = make_vae(K=1, IW=True)
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        with pytest.raises(ValueError, match="--model_name iwae"):
            model.calculate_loss((x, dummy_indices))
