"""
Unit tests for the IWAE model.

Tests cover:
  - Instantiation: resample flag, inheritance from VAE
  - K=1 equivalence with standard VAE
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

from models.IWAE import IWAE
from models.VAE import VAE


# ======================================================================================================================
# Helper: minimal args Namespace
# ======================================================================================================================
def make_args(**overrides):
    defaults = dict(
        model_name='iwae',
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


def make_iwae(K=5, **kwargs):
    args = make_args(K=K, **kwargs)
    return IWAE(args)


# ======================================================================================================================
# Instantiation
# ======================================================================================================================
class TestInstantiation:
    def test_resample_is_true(self):
        model = make_iwae(K=5)
        assert model.resample is True

    def test_resample_true_even_k1(self):
        with pytest.warns(UserWarning, match="K=1"):
            model = make_iwae(K=1)
        assert model.resample is True

    def test_inherits_from_vae(self):
        model = make_iwae()
        assert isinstance(model, VAE)

    def test_same_parameters_as_vae(self):
        """IWAE and VAE with same args should have identical parameter names."""
        iwae = make_iwae(K=5)
        vae_args = make_args(model_name='vae', K=1)
        vae = VAE(vae_args)
        assert set(iwae.state_dict().keys()) == set(vae.state_dict().keys())


# ======================================================================================================================
# K=1 equivalence
# ======================================================================================================================
class TestK1Equivalence:
    def test_k1_loss_matches_vae(self):
        """With K=1, IWAE loss should equal VAE loss (logsumexp of 1 element = element)."""
        torch.manual_seed(42)
        with pytest.warns(UserWarning, match="K=1"):
            iwae = make_iwae(K=1)

        vae_args = make_args(model_name='vae', K=1)
        vae = VAE(vae_args)

        # copy weights
        vae.load_state_dict(iwae.state_dict())

        x = torch.rand(8, np.prod([1, 28, 28]))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)

        torch.manual_seed(0)
        iwae_loss, iwae_re, iwae_kl = iwae.calculate_loss((x, dummy_indices), average=True)

        torch.manual_seed(0)
        vae_loss, vae_re, vae_kl = vae.calculate_loss((x, dummy_indices), average=True)

        assert torch.allclose(iwae_loss, vae_loss, atol=1e-5), \
            f"IWAE loss {iwae_loss.item():.6f} != VAE loss {vae_loss.item():.6f}"


# ======================================================================================================================
# Output shapes
# ======================================================================================================================
class TestOutputShapes:
    def test_no_average_shapes(self):
        model = make_iwae(K=5)
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=False)
        assert loss.shape == (8,)
        assert RE.shape == (8,)
        assert KL.shape == (8,)

    def test_average_shapes(self):
        model = make_iwae(K=5)
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
        model = make_iwae(K=5)
        x = torch.rand(4, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(4, 1, dtype=torch.long)
        loss, _, _ = model.calculate_loss((x, dummy_indices), average=True)
        loss.backward()

        # check encoder and decoder have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ======================================================================================================================
# Prior compatibility
# ======================================================================================================================
class TestPriorCompatibility:
    def test_vampprior_works(self):
        model = make_iwae(K=5, prior='vampprior')
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_exemplar_prior_raises(self):
        model = make_iwae(K=5, prior='exemplar_prior')
        x = torch.rand(8, np.prod(model.args.input_size))
        dummy_indices = torch.zeros(8, 1, dtype=torch.long)
        with pytest.raises(NotImplementedError, match="exemplar_prior"):
            model.calculate_loss((x, dummy_indices))
