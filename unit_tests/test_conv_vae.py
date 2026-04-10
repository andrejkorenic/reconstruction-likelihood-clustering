"""
Unit tests for ConvVAE model (TDD — written before implementation).

Tests cover:
  - GatedConvTranspose2d: spatial upsampling, no_attention mode
  - ConvVAE forward pass: output shapes for MNIST and RGB inputs
  - Loss: scalar output when averaged
  - Gradient flow: all parameters receive gradients
  - Training: loss decreases over a few SGD steps
  - Save/load: state_dict roundtrip produces identical outputs
"""

import argparse
import math

import pytest
import torch
import numpy as np

from utils.nn import GatedConvTranspose2d


# ======================================================================================================================
# Helper: minimal args Namespace
# ======================================================================================================================
def make_args(**overrides):
    defaults = dict(
        model_name='convvae',
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


def make_convvae(**kwargs):
    from models.ConvVAE import ConvVAE
    args = make_args(**kwargs)
    return ConvVAE(args)


# ======================================================================================================================
# GatedConvTranspose2d
# ======================================================================================================================
class TestGatedConvTranspose2d:
    def test_spatial_dims_doubled(self):
        """GatedConvTranspose2d with stride=2 should double spatial dimensions."""
        layer = GatedConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        x = torch.randn(2, 64, 7, 7)
        out = layer(x)
        assert out.shape == (2, 32, 14, 14), f"Expected (2, 32, 14, 14), got {out.shape}"

    def test_no_attention_mode(self):
        """no_attention=True should produce the same output shape (gating disabled)."""
        layer = GatedConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1, no_attention=True)
        x = torch.randn(2, 8, 7, 7)
        out = layer(x)
        assert out.shape == (2, 4, 14, 14), f"Expected (2, 4, 14, 14), got {out.shape}"


# ======================================================================================================================
# ConvVAE forward shape
# ======================================================================================================================
class TestConvVAEShape:
    def test_forward_output_shape(self):
        """forward() should return x_mean of shape (B, 784) and z_q_mean of shape (B, z1_size)."""
        model = make_convvae()
        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        x_mean, x_logvar, (z_q, z_q_mean, z_q_logvar) = model.forward(x)

        assert x_mean.shape == (B, 784), f"x_mean shape {x_mean.shape} != (B, 784)"
        assert z_q_mean.shape == (B, 4), f"z_q_mean shape {z_q_mean.shape} != (B, 4)"

    def test_loss_is_scalar_when_averaged(self):
        """calculate_loss with average=True should return scalar loss, RE, KL."""
        model = make_convvae()
        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        dummy_indices = torch.zeros(B, 1, dtype=torch.long)

        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)

        assert loss.dim() == 0, f"loss should be scalar, got dim={loss.dim()}"
        assert RE.dim() == 0, f"RE should be scalar, got dim={RE.dim()}"
        assert KL.dim() == 0, f"KL should be scalar, got dim={KL.dim()}"


# ======================================================================================================================
# Gradient flow
# ======================================================================================================================
class TestConvVAEGradients:
    def test_gradients_flow_to_all_parameters(self):
        """After loss.backward(), every requires_grad parameter should have .grad != None."""
        model = make_convvae()
        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        dummy_indices = torch.zeros(B, 1, dtype=torch.long)

        loss, _, _ = model.calculate_loss((x, dummy_indices), average=True)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ======================================================================================================================
# Training: loss decreases
# ======================================================================================================================
class TestConvVAETraining:
    def test_loss_decreases_over_5_steps(self):
        # Checks that step-5 loss is lower than step-1 (not necessarily monotone).
        """5 SGD steps should reduce the loss."""
        torch.manual_seed(42)
        model = make_convvae()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        B = 8
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        dummy_indices = torch.zeros(B, 1, dtype=torch.long)

        # record loss at step 1
        loss_first, _, _ = model.calculate_loss((x, dummy_indices), average=True)
        loss_first_val = loss_first.item()

        loss_first.backward()
        optimizer.step()
        optimizer.zero_grad()

        # run 4 more steps
        for _ in range(4):
            loss, _, _ = model.calculate_loss((x, dummy_indices), average=True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_last, _, _ = model.calculate_loss((x, dummy_indices), average=True)
        assert loss_last.item() < loss_first_val, \
            f"Loss did not decrease: {loss_first_val:.4f} -> {loss_last.item():.4f}"


# ======================================================================================================================
# Save / Load roundtrip
# ======================================================================================================================
class TestConvVAESaveLoad:
    def test_state_dict_roundtrip(self):
        """Save and reload state_dict — outputs should match exactly."""
        torch.manual_seed(42)
        model = make_convvae()

        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))

        torch.manual_seed(0)
        out_before, _, (_, _, _) = model.forward(x)

        # save and reload
        state = model.state_dict()
        model2 = make_convvae()
        model2.load_state_dict(state)

        torch.manual_seed(0)
        out_after, _, (_, _, _) = model2.forward(x)

        assert torch.allclose(out_before, out_after, atol=1e-6), \
            "Outputs differ after state_dict roundtrip"


# ======================================================================================================================
# RGB input (3, 32, 32)
# ======================================================================================================================
class TestConvVAERGBInput:
    def test_rgb_32x32_forward_shape(self):
        """ConvVAE with input_size=[3, 32, 32] should output x_mean of shape (B, 3072)."""
        model = make_convvae(input_size=[3, 32, 32], input_type='binary')
        B = 4
        x = torch.rand(B, 3 * 32 * 32)
        x_mean, x_logvar, (z_q, z_q_mean, z_q_logvar) = model.forward(x)

        assert x_mean.shape == (B, 3 * 32 * 32), \
            f"x_mean shape {x_mean.shape} != ({B}, 3072)"
        assert z_q_mean.shape == (B, 4), \
            f"z_q_mean shape {z_q_mean.shape} != ({B}, 4)"


# ======================================================================================================================
# Gray / Continuous input — use_logit branching
# ======================================================================================================================
class TestConvVAEGrayContinuous:
    """Tests for the use_logit clamping logic in ConvVAE.p_x()."""

    def test_gray_no_logit_output_clamped(self):
        """With use_logit=False, x_mean should be clamped within (1/512, 1-1/512)."""
        torch.manual_seed(0)
        model = make_convvae(input_type='gray', use_logit=False)
        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        x_mean, x_logvar, _ = model.forward(x)

        assert x_mean.min() >= 1. / 512., \
            f"x_mean min {x_mean.min().item()} < 1/512"
        assert x_mean.max() <= 1. - 1. / 512., \
            f"x_mean max {x_mean.max().item()} > 1 - 1/512"

    def test_gray_logit_output_unclamped(self):
        """With use_logit=True, x_mean is NOT clamped — p_x() returns raw decoder output."""
        torch.manual_seed(0)
        model = make_convvae(input_type='gray', use_logit=True)
        # Feed a large-magnitude latent directly into p_x to force extreme outputs
        z = torch.randn(4, model.args.z1_size) * 10.0
        x_mean, x_logvar = model.p_x(z)

        # With large z, the unconstrained decoder should produce values outside
        # the (1/512, 1-1/512) range that clamping would enforce.
        clamped = torch.clamp(x_mean, min=1./512., max=1.-1./512.)
        assert not torch.equal(x_mean, clamped), \
            "x_mean should not be identical to its clamped version in logit mode"

    def test_gray_no_logit_uses_decoder_logstd(self):
        """With use_logit=False, x_logvar should be uniform (from scalar decoder_logstd)."""
        torch.manual_seed(0)
        model = make_convvae(input_type='gray', use_logit=False)
        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        x_mean, x_logvar, _ = model.forward(x)

        # All pixels in each sample should have the same log-variance
        for i in range(B):
            unique_vals = x_logvar[i].unique()
            assert len(unique_vals) == 1, \
                f"Expected uniform x_logvar, got {len(unique_vals)} unique values"

    def test_gray_logit_uses_per_pixel_logvar(self):
        """With use_logit=True, x_logvar should vary across pixels (from conv head)."""
        torch.manual_seed(0)
        model = make_convvae(input_type='gray', use_logit=True)
        B = 4
        x = torch.randn(B, int(np.prod([1, 28, 28])))
        x_mean, x_logvar, _ = model.forward(x)

        # Per-pixel log-variance from the conv head should NOT be uniform
        for i in range(B):
            unique_vals = x_logvar[i].unique()
            assert len(unique_vals) > 1, \
                f"Expected per-pixel x_logvar variation, got {len(unique_vals)} unique values"

    def test_loss_finite_gray(self):
        """Full forward + loss with gray input should produce finite values."""
        torch.manual_seed(0)
        model = make_convvae(input_type='gray', use_logit=False)
        B = 4
        x = torch.rand(B, int(np.prod([1, 28, 28])))
        dummy_indices = torch.zeros(B, 1, dtype=torch.long)

        loss, RE, KL = model.calculate_loss((x, dummy_indices), average=True)

        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert torch.isfinite(RE), f"RE is not finite: {RE.item()}"
        assert torch.isfinite(KL), f"KL is not finite: {KL.item()}"
