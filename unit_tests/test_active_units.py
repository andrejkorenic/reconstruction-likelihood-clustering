"""Tests for Active Units computation and monitoring."""

import argparse

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.evaluation import compute_mean_variance_per_dimension


class FakeEncoder(nn.Module):
    """Encoder that returns fixed means for testing AU computation."""

    def __init__(self, means_matrix):
        super().__init__()
        self.means_matrix = means_matrix
        self._idx = 0

    def q_z(self, x, **kwargs):
        n = x.shape[0]
        start = self._idx
        self._idx += n
        means = self.means_matrix[start:start + n]
        logvar = torch.zeros_like(means)
        return means, logvar


def _make_loader(n_samples, input_dim, batch_size=10):
    """Create a simple (data, target) DataLoader."""
    data = torch.randn(n_samples, input_dim)
    targets = torch.zeros(n_samples, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(data, targets)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def test_compute_active_units_returns_count_and_variances():
    """Function returns (count, variances_array) tuple."""
    means = torch.zeros(20, 4)
    means[:, 0] = torch.linspace(-2, 2, 20)
    means[:, 1] = torch.linspace(-1, 1, 20)
    means[:, 2] = torch.full((20,), 0.5)
    means[:, 3] = torch.full((20,), -0.1)

    model = FakeEncoder(means)
    loader = _make_loader(20, 784, batch_size=10)

    count, variances = compute_mean_variance_per_dimension(
        model, loader, device=torch.device('cpu'), threshold=0.01
    )

    assert count == 2
    assert isinstance(variances, np.ndarray)
    assert variances.shape == (4,)
    assert variances[0] > 0.01
    assert variances[1] > 0.01
    assert variances[2] < 0.01
    assert variances[3] < 0.01


def test_compute_active_units_custom_threshold():
    """Higher threshold means fewer dims count as active."""
    means = torch.zeros(20, 3)
    means[:, 0] = torch.linspace(-2, 2, 20)
    means[:, 1] = torch.linspace(-0.1, 0.1, 20)
    means[:, 2] = torch.full((20,), 0.0)

    model = FakeEncoder(means)
    loader = _make_loader(20, 784, batch_size=20)

    count_low, _ = compute_mean_variance_per_dimension(
        model, loader, device=torch.device('cpu'), threshold=0.001
    )
    model._idx = 0
    count_high, _ = compute_mean_variance_per_dimension(
        model, loader, device=torch.device('cpu'), threshold=0.01
    )

    assert count_low == 2
    assert count_high == 1


from utils.active_units import ActiveUnitsMonitor


# ======================================================================
# ActiveUnitsMonitor tests
# ======================================================================
class TestActiveUnitsMonitor:

    def test_should_check_respects_interval(self):
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=3)
        assert not monitor.should_check(epoch=5, warmup=100)
        assert not monitor.should_check(epoch=50, warmup=100)
        assert monitor.should_check(epoch=105, warmup=100)
        assert not monitor.should_check(epoch=103, warmup=100)
        assert monitor.should_check(epoch=110, warmup=100)

    def test_should_check_disabled(self):
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=3)
        monitor.disable()
        assert not monitor.should_check(epoch=200, warmup=0)

    def test_is_stable_not_enough_readings(self):
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=5)
        monitor.record(40)
        monitor.record(38)
        monitor.record(36)
        assert not monitor.is_stable()

    def test_is_stable_all_same(self):
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=3)
        monitor.record(35)
        monitor.record(35)
        monitor.record(35)
        assert monitor.is_stable()

    def test_is_stable_within_tolerance(self):
        """max - min <= 1 counts as stable."""
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=3)
        monitor.record(35)
        monitor.record(36)
        monitor.record(35)
        assert monitor.is_stable()

    def test_is_stable_outside_tolerance(self):
        """max - min > 1 is not stable."""
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=3)
        monitor.record(35)
        monitor.record(37)
        monitor.record(35)
        assert not monitor.is_stable()

    def test_is_stable_uses_last_n_only(self):
        """Older unstable readings don't affect current stability window."""
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=3)
        monitor.record(50)
        monitor.record(45)
        monitor.record(40)
        monitor.record(35)
        monitor.record(35)
        monitor.record(35)
        assert monitor.is_stable()

    def test_get_result(self):
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=3)
        monitor.record(40)
        monitor.record(35)
        monitor.record(35)

        result = monitor.get_result(original_z1_size=128, epoch=25, user_accepted=True)

        assert result['discovered_z1_size'] == 35
        assert result['original_z1_size'] == 128
        assert result['stable_at_epoch'] == 25
        assert result['au_history'] == [40, 35, 35]
        assert result['user_accepted'] is True

    def test_latest_active_count(self):
        monitor = ActiveUnitsMonitor(check_interval=5, stability_count=3)
        monitor.record(40)
        monitor.record(35)
        assert monitor.latest == 35


import subprocess
import sys


def test_auto_z_size_args_are_parsed():
    """run.py accepts --auto_z_size and related flags without error."""
    result = subprocess.run(
        [sys.executable, 'run.py', '--help'],
        capture_output=True, text=True,
        cwd='/home/andrej/Documents/programiranje/new_vae_bundle',
    )
    assert '--auto_z_size' in result.stdout
    assert '--au_check_interval' in result.stdout
    assert '--au_stability_count' in result.stdout
    assert '--au_threshold' in result.stdout
