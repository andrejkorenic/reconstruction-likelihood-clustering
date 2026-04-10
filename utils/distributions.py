from __future__ import print_function
import torch
import torch.utils.data
import math

# ----- numeric stability constants -----

# Clamp bounds for probabilities passed to log(); avoids log(0) = -inf.
min_epsilon = 1e-5
max_epsilon = 1.-1e-5
log_sigmoid = torch.nn.LogSigmoid()
# Precomputed scalar used in every Gaussian log-density: log(2π)
log_2_pi = math.log(2*math.pi)


# ----- pairwise squared distance -----

def pairwise_distance(z, means):
    """Compute squared Euclidean distances between every row of z and every row of means.

    Uses the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b, which avoids
    materialising the full (MB, C, D) difference tensor and is much cheaper when D
    (latent dimension) is large.

    Computation is done in float64 to avoid catastrophic cancellation: the three
    terms can individually be large while their sum is small, so float32 precision
    is frequently insufficient.

    Args:
        z:     (MB, D) — batch of latent vectors
        means: (C,  D) — exemplar / component means

    Returns:
        (MB, C) float32 tensor of squared distances
    """
    # Upcast to float64 to avoid catastrophic cancellation in the ||a||^2 + ||b||^2 - 2ab expansion
    z = z.double()
    means = means.double()
    dist1 = (z**2).sum(dim=1).unsqueeze(1).expand(-1, means.shape[0])     # (MB, C) — squared norms of z rows, broadcast across components
    dist2 = (means**2).sum(dim=1).unsqueeze(0).expand(z.shape[0], -1)     # (MB, C) — squared norms of mean rows, broadcast across batch
    dist3 = torch.mm(z, torch.transpose(means, 0, 1))                     # (MB, C) — cross terms via a single matrix multiply
    # Downcast back to float32; results are now numerically stable
    return (dist1 + dist2 + - 2*dist3).float()


# ----- Gaussian log-densities -----

def log_normal_diag_vectorized(x, mean, log_var):
    """Log-density of a diagonal Gaussian evaluated for every (batch, component) pair.

    Used by the exemplar prior: x is a mini-batch of encoder outputs and mean is
    the full set of exemplar latent means, so we need all MB×C log-densities at once.

    The diagonal Gaussian log-density factors over dimensions as:
        log N(x; μ, σ²) = -½ Σ_d [log σ²_d + log(2π) + (x_d - μ_d)² / σ²_d]

    The squared Mahalanobis term is computed via pairwise_distance after
    whitening both x and mean by dividing by σ (= exp(log_var / 2)).  This
    avoids an explicit (MB, C, D) tensor while still being fully vectorised.

    Args:
        x:       (MB, D) — encoder samples z ~ q(z|x)
        mean:    (C,  D) — exemplar latent means, one per training point
        log_var: (C,  D) — exemplar log-variances (shared parameterisation)

    Returns:
        log_normal: (MB, C) — log N(x_i ; mean_c, diag(exp(log_var_c)))
        pair_dist:  (MB, C) — whitened squared distances (returned for reuse)
    """
    # σ_c for each component c; shape (C, D)
    log_var_sqrt = log_var.mul(0.5).exp_()
    # Whitening x and mean by σ turns the Mahalanobis distance into plain Euclidean
    pair_dist = pairwise_distance(x/log_var_sqrt, mean/log_var_sqrt)      # (MB, C)
    # Constant term: -½ Σ_d (log σ²_d + log 2π), summed over D, one value per component
    log_normal = -0.5 * torch.sum(log_var+log_2_pi, dim=1) - 0.5*pair_dist  # (MB, C)
    return log_normal, pair_dist


def log_normal_diag(x, mean, log_var, average=False, dim=None):
    """Log-density of a diagonal Gaussian, element-wise then reduced over dim.

    Implements: log N(x; μ, diag(σ²)) = -½ (log σ² + log(2π) + (x-μ)²/σ²)
    summed (or averaged) over the specified dimension.

    dim=1 is the typical choice during training: sums over the D latent
    dimensions to obtain a scalar per batch element, producing a (MB,) tensor.

    Args:
        x:       (..., D) — observed values
        mean:    (..., D) — distribution mean
        log_var: (..., D) — log of the diagonal variance
        average: if True, take mean instead of sum over dim
        dim:     dimension to reduce; None reduces all dims

    Returns:
        Scalar or (MB,) tensor depending on dim
    """
    # Element-wise log-density; shape matches inputs
    log_normal = -0.5 * (log_var + log_2_pi + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean(log_normal, dim)
    else:
        # Sum over latent dimensions (dim=1) gives per-sample log-likelihood
        return torch.sum(log_normal, dim)


def log_normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x, 2) - 0.5 * log_2_pi*x.new_ones(size=x.shape)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


# ----- discrete / bounded observation likelihoods -----

def log_bernoulli(x, mean, average=False, dim=None):
    """Log-likelihood under a Bernoulli distribution, for binary inputs.

    Implements: log p(x) = x log p + (1-x) log(1-p)

    Args:
        x:    (..., D) — binary observations in {0, 1}
        mean: (..., D) — predicted Bernoulli probabilities (decoder output,
                         typically after sigmoid)
        average: if True, take mean instead of sum over dim
        dim:     dimension to reduce

    Returns:
        Scalar or (MB,) tensor depending on dim
    """
    # Clamp predicted probabilities away from 0 and 1 before taking log;
    # without this, log(0) = -inf causes NaN gradients when mean saturates.
    probs = torch.clamp( mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)

    if average:
        return torch.mean(log_bernoulli, dim)
    else:
        return torch.sum(log_bernoulli, dim)


def log_logistic_256(x, mean, logvar, average=False, reduce=True, dim=None):
    """Log-likelihood under a discretized logistic distribution for 8-bit images.

    Models each pixel as a continuous logistic distribution that has been
    discretized into 256 uniform bins of width 1/256.  The probability of
    observing pixel value x is the CDF mass of the bin that x falls into:

        p(x) = σ((x_quant + bin_size - μ) / s) - σ((x_quant - μ) / s)

    where x_quant = floor(x / bin_size) * bin_size snaps x to the bin's lower
    edge, s = exp(logvar) is the scale, and σ is the sigmoid / logistic CDF.

    Reference: https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28

    Args:
        x:      (..., D) — pixel values in [0, 1] (normalized from [0, 255])
        mean:   (..., D) — predicted logistic means (decoder output)
        logvar: (..., D) — predicted log-scale of the logistic distribution
        average: if True, take mean instead of sum over dim
        reduce:  unused flag kept for API compatibility
        dim:     dimension to reduce

    Returns:
        Scalar or (MB,) tensor depending on dim
    """
    # Each of the 256 pixel levels occupies a bin of this width in [0, 1]
    bin_size = 1. / 256.
    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    # Quantize x to the lower edge of its bin, then shift and scale to logistic coordinates
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    # CDF evaluated at the upper edge of the bin (x already points to lower edge)
    cdf_plus = torch.sigmoid(x + bin_size/scale)
    # CDF evaluated at the lower edge of the bin
    cdf_minus = torch.sigmoid(x)
    # 1e-7 guards against log(0) when cdf_plus ≈ cdf_minus (e.g. very small scale)
    log_logist_256 = torch.log(cdf_plus - cdf_minus + 1e-7)

    if average:
        return torch.mean(log_logist_256, dim)
    else:
        return torch.sum(log_logist_256, dim)


def log_beta(x, mean, log_concentration, average=False, dim=None):
    """Log-probability under Beta distribution (mean-concentration parameterization).

    Reparameterises the standard Beta(α, β) in terms of:
        μ = α / (α + β)    (mean, in (0, 1))
        ν = α + β          (concentration / total count, > 0)
    so that α = μν and β = (1-μ)ν.

    Log-density (up to constants):
        log p(x) = (α-1) log x + (β-1) log(1-x)
                   - log Γ(α) - log Γ(β) + log Γ(α+β)

    Args:
        x:                 (..., D) — observed values in (0, 1)
        mean:              (..., D) — predicted mean μ ∈ (0, 1) — typically sigmoid output
        log_concentration: (..., D) — log(ν) where ν > 0 is the concentration
        average: if True, take mean instead of sum over dim
        dim:     dimension to reduce

    Returns:
        Scalar or (MB,) tensor depending on dim
    """
    eps = 1e-6
    # Clamp inputs and means away from the boundaries where log(x) → -inf
    x = x.clamp(eps, 1.0 - eps)
    mean = mean.clamp(eps, 1.0 - eps)

    nu = torch.exp(log_concentration)    # concentration ν = α + β, shape (..., D)
    alpha = mean * nu                    # α = μν,       shape (..., D)
    beta_ = (1.0 - mean) * nu           # β = (1-μ)ν,   shape (..., D)

    log_prob = (
        (alpha - 1.0) * torch.log(x)
        + (beta_ - 1.0) * torch.log(1.0 - x)
        - torch.lgamma(alpha)
        - torch.lgamma(beta_)
        + torch.lgamma(alpha + beta_)   # normalisation constant log B(α,β)^{-1}
    )

    if average:
        return torch.mean(log_prob, dim)
    else:
        return torch.sum(log_prob, dim)

