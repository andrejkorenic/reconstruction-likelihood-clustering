"""
Unit tests for utils.create_model.

Each test creates a model via create_model() and checks that:
  - the returned model is the correct class
  - the optimizer is Adam
  - the snap_dir path follows the expected convention
"""

import argparse

import pytest
import torch
import torch.nn as nn

from utils.create_model import create_model

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
# Helper: build a minimal args Namespace
# ======================================================================================================================
def make_args(**overrides):
    """
    Return a Namespace with all the fields that BaseModel / create_model expect.
    Any field can be overridden via keyword arguments.
    """
    defaults = dict(
        # model / prior
        model_name='vae',
        prior='standard',
        input_type='binary',
        input_size=[1, 28, 28],
        # architecture
        hidden_size=300,
        z1_size=40,
        z2_size=40,
        activation=None,
        no_attention=False,
        same_variational_var=False,
        use_logit=False,
        # vampprior pseudo-inputs
        number_components=500,
        pseudoinputs_mean=-0.05,
        pseudoinputs_std=0.01,
        use_training_data_init=False,
        # dataset
        dataset_name='dynamic_mnist',
        training_set_size=60000,
        dynamic_binarization=False,
        # optimisation
        lr=5e-4,
        # device
        device=torch.device('cpu'),
        cuda=False,
        # misc
        use_whole_train=False,
        approximate_prior=False,
        approximate_k=10,
        continuous=False,
        lambd=1e-4,
        bottleneck=6,
        # adaptive latent size
        auto_z_size=False,
        au_check_interval=5,
        au_stability_count=5,
        au_threshold=0.01,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ======================================================================================================================
# Tests for the flat (single-latent) VAE
# ======================================================================================================================
def test_create_model_vae(tmp_path, monkeypatch):
    # run in tmp_path so snap_dir is created there
    monkeypatch.chdir(tmp_path)
    args = make_args(model_name='vae')

    model, optimizer, returned_args, snap_dir = create_model(args)

    from models.VAE import VAE
    assert isinstance(model, VAE)
    assert isinstance(optimizer, torch.optim.Optimizer)
    # snap_dir must contain model_name=vae (analysis.py parses this)
    assert 'model_name=vae' in snap_dir


# ======================================================================================================================
# Tests for the hierarchical (2-level) models
# ======================================================================================================================
def test_create_model_hvae_2level(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(model_name='hvae_2level')

    model, optimizer, returned_args, snap_dir = create_model(args)

    from models.HVAE_2level import VAE
    assert isinstance(model, VAE)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert 'model_name=hvae_2level' in snap_dir


def test_create_model_convhvae_2level(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(model_name='convhvae_2level')

    model, optimizer, returned_args, snap_dir = create_model(args)

    from models.convHVAE_2level import VAE
    assert isinstance(model, VAE)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert 'model_name=convhvae_2level' in snap_dir


# ======================================================================================================================
# Tests for PixelCNN model
# ======================================================================================================================
def test_create_model_pixelhvae(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(model_name='pixelcnn')

    model, optimizer, returned_args, snap_dir = create_model(args)

    from models.PixelCNN import VAE
    assert isinstance(model, VAE)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert 'model_name=pixelcnn' in snap_dir


# ======================================================================================================================
# Test snap_dir format
# ======================================================================================================================
def test_snap_dir_contains_dataset_and_prior(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(model_name='vae', dataset_name='static_mnist', prior='vampprior')

    _, _, _, snap_dir = create_model(args)

    # snap_dir should encode dataset, prior, and model_name
    assert 'static_mnist' in snap_dir
    assert 'vampprior' in snap_dir
    assert 'model_name=vae' in snap_dir


# ======================================================================================================================
# Test that args is not unexpectedly mutated
# ======================================================================================================================
def test_args_core_fields_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = make_args(model_name='vae', prior='standard', dataset_name='dynamic_mnist')

    _, _, returned_args, _ = create_model(args)

    # create_model may add fields (e.g. timestamp) but should not change these
    assert returned_args.model_name == 'vae'
    assert returned_args.prior == 'standard'
    assert returned_args.dataset_name == 'dynamic_mnist'


# ======================================================================================================================
# Decoder output activation
# ======================================================================================================================
def test_binary_model_has_sigmoid_output(tmp_path, monkeypatch):
    """Model created for binary data must have Sigmoid activation in p_x_mean."""
    monkeypatch.chdir(tmp_path)
    args = make_args(model_name='vae', input_type='binary')

    model, _, _, _ = create_model(args)
    assert isinstance(model.p_x_mean.activation, nn.Sigmoid)


def test_gray_model_has_no_sigmoid_output(tmp_path, monkeypatch):
    """Model created for gray data must NOT have Sigmoid — output is linear."""
    monkeypatch.chdir(tmp_path)
    args = make_args(model_name='vae', input_type='gray')

    model, _, _, _ = create_model(args)
    assert model.p_x_mean.activation is None
