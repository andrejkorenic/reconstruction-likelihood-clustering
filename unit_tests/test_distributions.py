"""
Unit tests for reconstruction likelihood functions in utils/distributions.py.

Verifies correctness (PyTorch vs NumPy reference), sign convention (all functions
return log p(x), not NLL), edge-case stability, and gradient propagation.

Ported from vae-reconstruction-probability/utils/distributions.py __main__ block,
adapted for our sign convention (log p(x) everywhere, not -log p(x)).
"""

import math
import pytest
import numpy as np
import numpy.testing as npt
import torch

from utils.distributions import (
    log_normal_diag,
    log_normal_standard,
    log_bernoulli,
    log_logistic_256,
    log_beta,
)

# ===================================================================
# NumPy reference implementations
# ===================================================================

LOG_2_PI = math.log(2 * math.pi)


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def ref_log_normal_diag_np(x, mean, log_var):
    """Reference Gaussian log-density, summed over last dim."""
    per_elem = -0.5 * (log_var + LOG_2_PI + (x - mean) ** 2 / np.exp(log_var))
    return per_elem.sum(axis=-1)


def ref_log_bernoulli_np(x, mean, eps=1e-5):
    """Reference Bernoulli log-likelihood, summed over last dim."""
    p = np.clip(mean, eps, 1.0 - eps)
    per_elem = x * np.log(p) + (1.0 - x) * np.log(1.0 - p)
    return per_elem.sum(axis=-1)


def ref_log_logistic_256_np(x, mean, logvar, bin_size=1.0 / 256.0, eps=1e-7):
    """Reference discretized logistic log-likelihood, summed over last dim.

    Returns log p(x) — positive convention (matches our code, NOT the reference repo).
    """
    scale = np.exp(logvar)
    x_floor = np.floor(x / bin_size) * bin_size
    z = (x_floor - mean) / scale
    cdf_plus = _sigmoid_np(z + bin_size / scale)
    cdf_minus = _sigmoid_np(z)
    log_prob = np.log(cdf_plus - cdf_minus + eps)  # log p(x) — no negation
    return log_prob.sum(axis=-1)


def ref_log_beta_np(x, mean, log_concentration, eps=1e-6):
    """Reference Beta log-density (mean-concentration), summed over last dim."""
    from scipy.special import gammaln
    x = np.clip(x, eps, 1.0 - eps)
    mean = np.clip(mean, eps, 1.0 - eps)
    nu = np.exp(log_concentration)
    alpha = mean * nu
    beta_ = (1.0 - mean) * nu
    log_prob = (
        (alpha - 1.0) * np.log(x)
        + (beta_ - 1.0) * np.log(1.0 - x)
        - gammaln(alpha)
        - gammaln(beta_)
        + gammaln(alpha + beta_)
    )
    return log_prob.sum(axis=-1)


# ===================================================================
# Sign convention: all functions must return log p(x), not -log p(x)
# ===================================================================

class TestSignConvention:
    """All distribution functions should return log-likelihood (negative for most inputs)."""

    def test_gaussian_sign(self):
        """Gaussian log-density should be negative for typical inputs."""
        x = torch.rand(4, 10)
        mean = torch.rand(4, 10)
        logvar = torch.zeros(4, 10)
        result = log_normal_diag(x, mean, logvar, dim=1)
        # log N(x; mu, 1) is always negative (density < 1 for d > 1)
        assert (result < 0).all(), "Gaussian log-density should be negative for d=10"

    def test_logistic_256_sign(self):
        """Discretized logistic log-prob should be negative (probability < 1)."""
        x = torch.rand(4, 784)
        mean = torch.rand(4, 784)
        logvar = torch.zeros(4, 784)
        result = log_logistic_256(x, mean, logvar, dim=1)
        assert (result < 0).all(), "log_logistic_256 should return log p(x) < 0"

    def test_bernoulli_sign(self):
        """Bernoulli log-prob should be negative."""
        x = torch.round(torch.rand(4, 784))
        mean = torch.rand(4, 784).clamp(0.01, 0.99)
        result = log_bernoulli(x, mean, dim=1)
        assert (result < 0).all(), "log_bernoulli should return log p(x) < 0"

    def test_beta_sign(self):
        """Beta log-density can be positive (density > 1 near mode), but should be finite."""
        x = torch.rand(4, 10).clamp(0.1, 0.9)
        mean = torch.rand(4, 10).clamp(0.1, 0.9)
        log_conc = torch.ones(4, 10) * 2.0  # moderate concentration
        result = log_beta(x, mean, log_conc, dim=1)
        assert torch.isfinite(result).all()

    def test_standard_normal_sign(self):
        """Standard normal log-density should be negative for d > 1."""
        x = torch.randn(4, 40)
        result = log_normal_standard(x, dim=1)
        assert (result < 0).all()


# ===================================================================
# PyTorch vs NumPy reference: correctness
# ===================================================================

class TestGaussianCorrectness:
    def test_random_values(self):
        torch.manual_seed(42)
        x = torch.randn(8, 20)
        mean = torch.randn(8, 20)
        logvar = (torch.rand(8, 20) - 0.5) * 2.0

        out_torch = log_normal_diag(x, mean, logvar, dim=1)
        out_np = ref_log_normal_diag_np(x.numpy(), mean.numpy(), logvar.numpy())

        npt.assert_allclose(out_torch.numpy(), out_np, rtol=1e-5, atol=1e-5)

    def test_zero_variance(self):
        """logvar=0 means sigma=1 — standard normal around the mean."""
        x = torch.tensor([[0.0, 0.0]])
        mean = torch.tensor([[0.0, 0.0]])
        logvar = torch.tensor([[0.0, 0.0]])

        result = log_normal_diag(x, mean, logvar, dim=1)
        expected = -0.5 * 2 * LOG_2_PI  # two dims of N(0;0,1)
        npt.assert_allclose(result.item(), expected, rtol=1e-5)


class TestLogistic256Correctness:
    def test_random_values(self):
        """Compare PyTorch vs NumPy for random inputs (flat tensors, dim=1)."""
        torch.manual_seed(42)
        x = torch.rand(8, 784)
        mean = torch.rand(8, 784)
        logvar = (torch.rand(8, 784) - 0.5) * 2.0

        out_torch = log_logistic_256(x, mean, logvar, dim=1)
        out_np = ref_log_logistic_256_np(x.numpy(), mean.numpy(), logvar.numpy())

        assert out_torch.shape == (8,)
        npt.assert_allclose(out_torch.numpy(), out_np, rtol=1e-5, atol=1e-5)

    def test_edge_cases(self):
        """Values on bin boundaries and extremes (0.0, 1.0)."""
        bin_size = 1.0 / 256.0
        x_vals = np.array([[0.0, bin_size - 1e-9, bin_size, 0.5, 0.999, 1.0 - 1e-9]],
                          dtype=np.float32)
        mean_np = np.zeros_like(x_vals)
        logvar_np = np.log(np.ones_like(x_vals) * 0.1)

        x = torch.tensor(x_vals)
        mean = torch.tensor(mean_np)
        logvar = torch.tensor(logvar_np)

        out_torch = log_logistic_256(x, mean, logvar, dim=1)
        out_np = ref_log_logistic_256_np(x_vals, mean_np, logvar_np)

        npt.assert_allclose(out_torch.numpy(), out_np, rtol=1e-5, atol=1e-6)


class TestBernoulliCorrectness:
    def test_random_values(self):
        torch.manual_seed(42)
        x = torch.round(torch.rand(8, 784))
        mean = torch.rand(8, 784).clamp(0.01, 0.99)

        out_torch = log_bernoulli(x, mean, dim=1)
        out_np = ref_log_bernoulli_np(x.numpy(), mean.numpy())

        npt.assert_allclose(out_torch.numpy(), out_np, rtol=1e-5, atol=1e-5)

    def test_perfect_prediction(self):
        """When mean exactly matches x, log-prob should be close to 0."""
        x = torch.tensor([[1.0, 0.0, 1.0]])
        mean = torch.tensor([[0.999, 0.001, 0.999]])
        result = log_bernoulli(x, mean, dim=1)
        assert result.item() > -0.01  # very close to 0


class TestBetaCorrectness:
    def test_random_values(self):
        torch.manual_seed(42)
        x = torch.rand(8, 140).clamp(0.01, 0.99)
        mean = torch.rand(8, 140).clamp(0.01, 0.99)
        log_conc = torch.ones(8, 140) * 3.0

        out_torch = log_beta(x, mean, log_conc, dim=1)
        out_np = ref_log_beta_np(x.numpy(), mean.numpy(), log_conc.numpy())

        npt.assert_allclose(out_torch.numpy(), out_np, rtol=1e-4, atol=1e-4)

    def test_mode_concentration(self):
        """Higher concentration near the mean should give higher log-prob."""
        x = torch.tensor([[0.5, 0.5]])
        mean = torch.tensor([[0.5, 0.5]])
        low = log_beta(x, mean, torch.tensor([[2.0, 2.0]]), dim=1)
        high = log_beta(x, mean, torch.tensor([[5.0, 5.0]]), dim=1)
        assert high > low


# ===================================================================
# Edge cases and numerical stability
# ===================================================================

class TestNumericalStability:
    def test_bernoulli_saturated_predictions(self):
        """mean=0 and mean=1 should not produce NaN."""
        x = torch.tensor([[1.0, 0.0]])
        mean = torch.tensor([[0.0, 1.0]])  # worst case: opposite of x
        result = log_bernoulli(x, mean, dim=1)
        assert torch.isfinite(result).all()

    def test_logistic_extreme_logvar(self):
        """Very large/small logvar should not produce NaN."""
        x = torch.rand(2, 10)
        mean = torch.rand(2, 10)
        for lv in [-10.0, 0.0, 10.0]:
            logvar = torch.full((2, 10), lv)
            result = log_logistic_256(x, mean, logvar, dim=1)
            assert torch.isfinite(result).all(), f"NaN/Inf at logvar={lv}"

    def test_beta_boundary_values(self):
        """x=0 and x=1 should not produce NaN (clamping protects)."""
        x = torch.tensor([[0.0, 0.5, 1.0]])
        mean = torch.tensor([[0.5, 0.5, 0.5]])
        log_conc = torch.ones(1, 3) * 3.0
        result = log_beta(x, mean, log_conc, dim=1)
        assert torch.isfinite(result).all()

    def test_gaussian_large_deviation(self):
        """Large x-mean distance should give very negative log-prob, not NaN."""
        x = torch.tensor([[100.0]])
        mean = torch.tensor([[0.0]])
        logvar = torch.tensor([[0.0]])
        result = log_normal_diag(x, mean, logvar, dim=1)
        assert torch.isfinite(result).all()
        assert result.item() < -1000


# ===================================================================
# Gradient propagation
# ===================================================================

class TestGradients:
    @pytest.mark.parametrize("fn,make_inputs", [
        ("gaussian", lambda: (
            torch.rand(4, 20),
            torch.randn(4, 20, requires_grad=True),
            torch.randn(4, 20, requires_grad=True),
        )),
        ("logistic", lambda: (
            torch.rand(4, 784),
            torch.randn(4, 784, requires_grad=True),
            torch.randn(4, 784, requires_grad=True),
        )),
        ("beta", lambda: (
            torch.rand(4, 140).clamp(0.01, 0.99),
            torch.rand(4, 140, requires_grad=True),
            torch.full((4, 140), 3.0, requires_grad=True),
        )),
    ])
    def test_gradients_flow(self, fn, make_inputs):
        """Gradients should flow through mean and logvar/concentration."""
        x, param1, param2 = make_inputs()

        if fn == "gaussian":
            out = log_normal_diag(x, param1, param2, dim=1)
        elif fn == "logistic":
            out = log_logistic_256(x, param1, param2, dim=1)
        elif fn == "beta":
            out = log_beta(x, param1, param2, dim=1)

        out.sum().backward()

        assert param1.grad is not None, f"{fn}: no gradient for param1"
        assert param2.grad is not None, f"{fn}: no gradient for param2"
        assert torch.isfinite(param1.grad).all(), f"{fn}: non-finite param1 grad"
        assert torch.isfinite(param2.grad).all(), f"{fn}: non-finite param2 grad"
