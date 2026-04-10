"""
Unit tests for utils.load_data.

Tests that load_dataset() returns loaders with the correct tuple format,
sets runtime args correctly, and produces data with the expected shape.

Uses dynamic_mnist (auto-downloaded via torchvision).
"""

import argparse

import numpy as np
import pytest
import torch

from utils.load_data.data_loader_instances import load_dataset

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
# Helper: build a minimal args Namespace for data loading
# ======================================================================================================================
def make_args(**overrides):
    defaults = dict(
        dataset_name='dynamic_mnist',
        batch_size=100,
        test_batch_size=100,
        training_set_size=50000,
        input_size=[1, 28, 28],
        input_type='binary',
        dynamic_binarization=True,
        # pseudo-inputs
        number_components=500,
        pseudoinputs_mean=-0.05,
        pseudoinputs_std=0.01,
        use_training_data_init=False,
        # data preprocessing
        continuous=False,
        use_logit=False,
        lambd=1e-4,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ======================================================================================================================
# Test: train loader yields (data, indices, target) — 3-tuple
# ======================================================================================================================
def test_train_loader_is_3_tuple():
    args = make_args()
    train_loader, val_loader, test_loader, args = load_dataset(args)

    batch = next(iter(train_loader))
    assert len(batch) == 3, f"Expected 3-tuple (data, indices, target), got {len(batch)}-tuple"


# ======================================================================================================================
# Test: val/test loaders yield (data, target) — 2-tuple
# ======================================================================================================================
def test_val_loader_is_2_tuple():
    args = make_args()
    train_loader, val_loader, test_loader, args = load_dataset(args)

    batch = next(iter(val_loader))
    assert len(batch) == 2, f"Expected 2-tuple (data, target), got {len(batch)}-tuple"


def test_test_loader_is_2_tuple():
    args = make_args()
    train_loader, val_loader, test_loader, args = load_dataset(args)

    batch = next(iter(test_loader))
    assert len(batch) == 2, f"Expected 2-tuple (data, target), got {len(batch)}-tuple"


# ======================================================================================================================
# Test: runtime args set correctly for dynamic_mnist
# ======================================================================================================================
def test_dynamic_mnist_runtime_args():
    args = make_args()
    _, _, _, args = load_dataset(args)

    assert args.input_size == [1, 28, 28]
    assert args.input_type == 'binary'
    assert args.dynamic_binarization is True
    assert args.training_set_size == 50000


# ======================================================================================================================
# Test: data shape matches input_size (flat vector)
# ======================================================================================================================
def test_data_shape():
    args = make_args()
    train_loader, _, _, args = load_dataset(args)

    data, indices, target = next(iter(train_loader))
    expected_flat_size = int(np.prod(args.input_size))
    assert data.shape[1] == expected_flat_size, \
        f"Expected flat vector of size {expected_flat_size}, got {data.shape[1]}"


# ======================================================================================================================
# Test: continuous flag gives gray input_type
# ======================================================================================================================
def test_continuous_flag_gives_gray():
    args = make_args(continuous=True)
    _, _, _, args = load_dataset(args)

    assert args.input_type == 'gray'
    assert args.dynamic_binarization is False
