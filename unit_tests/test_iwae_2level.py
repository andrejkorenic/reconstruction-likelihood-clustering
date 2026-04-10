"""
Unit tests for the IWAE_2level model (two stochastic hidden layers).

Tests cover:
  - Instantiation: resample flag, inheritance from HVAE_2level
  - K=1 equivalence with standard HVAE_2level
  - Forward output shapes with K > 1
  - Output shapes for calculate_loss()
  - Gradient flow through IWAE loss
  - VampPrior compatibility
  - Exemplar prior guard
"""

import argparse
import math

import pytest
import torch
import numpy as np

from models.IWAE_2level import IWAE_2level
from models.HVAE_2level import VAE as HVAE_2level


# ======================================================================================================================
# Helper: minimal args Namespace
# ======================================================================================================================
def make_args(**overrides):
    defaults = dict(
        model_name='iwae_2level',
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
        K=5,
        IW=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_iwae_2level(K=5, **kwargs):
    args = make_args(K=K, **kwargs)
    return IWAE_2level(args)


# ======================================================================================================================
# Instantiation
# ======================================================================================================================
class TestInstantiation:
    def test_resample_is_false(self):
        model = make_iwae_2level(K=5)
        assert model.resample is False

    def test_resample_false_even_k1(self):
        with pytest.warns(UserWarning, match="K=1"):
            model = make_iwae_2level(K=1)
        assert model.resample is False

    def test_inherits_from_hvae_2level(self):
        model = make_iwae_2level()
        assert isinstance(model, HVAE_2level)

    def test_same_parameters_as_hvae_2level(self):
        """IWAE_2level and HVAE_2level with same args should have identical parameter names."""
        iwae = make_iwae_2level(K=5)
        hvae_args = make_args(model_name='hvae_2level', K=1)
        hvae = HVAE_2level(hvae_args)
        assert set(iwae.state_dict().keys()) == set(hvae.state_dict().keys())


# ======================================================================================================================
# K=1 equivalence
# ======================================================================================================================
class TestK1Equivalence:
    def test_k1_loss_matches_hvae(self):
        """With K=1, IWAE_2level loss should equal HVAE_2level loss."""
        torch.manual_seed(42)
        with pytest.warns(UserWarning, match="K=1"):
            iwae = make_iwae_2level(K=1)

        hvae_args = make_args(model_name='hvae_2level', K=1)
        hvae = HVAE_2level(hvae_args)

        # copy weights
        hvae.load_state_dict(iwae.state_dict())

        x = torch.rand(8, np.prod([1, 28, 28]))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)

        torch.manual_seed(0)
        iwae_loss, iwae_re, iwae_kl = iwae.calculate_loss(
            (x, dummy_indices), average=True)

        torch.manual_seed(0)
        hvae_loss, hvae_re, hvae_kl = hvae.calculate_loss(
            (x, dummy_indices), average=True)

        assert torch.allclose(iwae_loss, hvae_loss, atol=1e-5), \
            f"IWAE loss {iwae_loss.item():.6f} != HVAE loss {hvae_loss.item():.6f}"


# ======================================================================================================================
# Forward output shapes
# ======================================================================================================================
class TestForwardShapes:
    def test_forward_shapes_k5(self):
        model = make_iwae_2level(K=5)
        x = torch.rand(8, np.prod(model.args.input_size))
        x_mean, x_logvar, latent_stats = model.forward(x)

        K, batch = 5, 8
        z1_dim = model.args.z1_size
        z2_dim = model.args.z2_size
        D = np.prod(model.args.input_size)

        assert x_mean.shape == (K * batch, D)
        # z1_q, z1_q_mean, z1_q_logvar
        assert latent_stats[0].shape == (K * batch, z1_dim)
        assert latent_stats[1].shape == (K * batch, z1_dim)
        assert latent_stats[2].shape == (K * batch, z1_dim)
        # z2_q, z2_q_mean, z2_q_logvar
        assert latent_stats[3].shape == (K * batch, z2_dim)
        assert latent_stats[4].shape == (K * batch, z2_dim)
        assert latent_stats[5].shape == (K * batch, z2_dim)
        # z1_p_mean, z1_p_logvar
        assert latent_stats[6].shape == (K * batch, z1_dim)
        assert latent_stats[7].shape == (K * batch, z1_dim)


# ======================================================================================================================
# Output shapes
# ======================================================================================================================
class TestOutputShapes:
    def test_no_average_shapes(self):
        model = make_iwae_2level(K=5)
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=False)
        assert loss.shape == (8,)
        assert RE.shape == (8,)
        assert KL.shape == (8,)

    def test_average_shapes(self):
        model = make_iwae_2level(K=5)
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)
        assert loss.dim() == 0
        assert RE.dim() == 0
        assert KL.dim() == 0


# ======================================================================================================================
# Gradient flow
# ======================================================================================================================
class TestGradientFlow:
    def test_gradients_flow_to_parameters(self):
        model = make_iwae_2level(K=5)
        x = torch.rand(4, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(4, 1, dtype=torch.long)
        loss, _, _ = model.calculate_loss((x, dummy_indices), average=True)
        loss.backward()

        # check that all trainable parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ======================================================================================================================
# Prior compatibility
# ======================================================================================================================
class TestPriorCompatibility:
    def test_vampprior_works(self):
        model = make_iwae_2level(K=5, prior='vampprior')
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_exemplar_prior_raises(self):
        model = make_iwae_2level(K=5, prior='exemplar_prior')
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        with pytest.raises(NotImplementedError, match="exemplar_prior"):
            model.calculate_loss((x, dummy_indices))
