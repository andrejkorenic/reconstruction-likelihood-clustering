"""
Unit tests for convHVAE_2level model with transposed conv decoder.

Tests cover:
  - Forward pass output shapes for MNIST (1, 28, 28)
  - Loss produces finite values
  - Decoder spatial bottleneck size is correct
  - Forward pass works for CIFAR-like input (3, 32, 32)
"""

import argparse

import pytest
import torch
import numpy as np


# ======================================================================================================================
# Helper: minimal args Namespace (mirrors test_conv_vae.py pattern)
# ======================================================================================================================
def make_args(**overrides):
    defaults = dict(
        model_name='convhvae_2level',
        prior='standard',
        input_type='binary',
        input_size=[1, 28, 28],
        hidden_size=32,
        z1_size=4,
        z2_size=4,
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


def make_model(**kwargs):
    from models.convHVAE_2level import VAE
    args = make_args(**kwargs)
    return VAE(args)


# ======================================================================================================================
# Forward shape (MNIST)
# ======================================================================================================================
class TestForwardShape:
    def test_forward_output_shape(self):
        """forward() should return x_mean of shape (B, 784) for MNIST."""
        torch.manual_seed(42)
        model = make_model()
        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        x_mean, x_logvar, latent_stats = model.forward(x)

        assert x_mean.shape == (B, 784), f"x_mean shape {x_mean.shape} != (B, 784)"
        # latent_stats is 8-tuple for hierarchical model
        z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = latent_stats
        assert z1_q_mean.shape == (B, 4), f"z1_q_mean shape {z1_q_mean.shape} != (B, 4)"
        assert z2_q_mean.shape == (B, 4), f"z2_q_mean shape {z2_q_mean.shape} != (B, 4)"


# ======================================================================================================================
# Loss finite
# ======================================================================================================================
class TestLossFinite:
    def test_loss_is_finite(self):
        """calculate_loss should return finite loss, RE, KL."""
        torch.manual_seed(42)
        model = make_model()
        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        dummy_indices = torch.zeros(B, 1, dtype=torch.long)

        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert torch.isfinite(RE), f"RE is not finite: {RE.item()}"
        assert torch.isfinite(KL), f"KL is not finite: {KL.item()}"


# ======================================================================================================================
# Decoder spatial bottleneck
# ======================================================================================================================
class TestDecoderSpatialBottleneck:
    def test_decoder_h_size_mnist(self):
        """For MNIST (28x28), decoder_h_size should be 64 * 7 * 7 = 3136."""
        model = make_model()
        assert model.decoder_h_size == 64 * 7 * 7, \
            f"decoder_h_size {model.decoder_h_size} != {64 * 7 * 7}"

    def test_decoder_h_size_cifar(self):
        """For CIFAR (32x32), decoder_h_size should be 64 * 8 * 8 = 4096."""
        model = make_model(input_size=[3, 32, 32])
        assert model.decoder_h_size == 64 * 8 * 8, \
            f"decoder_h_size {model.decoder_h_size} != {64 * 8 * 8}"


# ======================================================================================================================
# Forward with CIFAR-like input (3, 32, 32)
# ======================================================================================================================
class TestForwardCIFAR:
    def test_forward_cifar(self):
        """convHVAE_2level with input_size=[3,32,32] should produce correct output shape."""
        torch.manual_seed(42)
        model = make_model(input_size=[3, 32, 32])
        B = 4
        x = torch.rand(B, 3 * 32 * 32)
        x_mean, x_logvar, latent_stats = model.forward(x)

        assert x_mean.shape == (B, 3 * 32 * 32), \
            f"x_mean shape {x_mean.shape} != ({B}, 3072)"
