"""
Importance Weighted Autoencoder (IWAE).

Inherits the VAE encoder/decoder architecture and overrides the loss
computation with the importance-weighted ELBO from Burda et al. (2015):

    L_K(x) = logsumexp(log w_1, ..., log w_K) - log K

where log w_k = log p(x|z^(k)) + log p(z^(k)) - log q(z^(k)|x).

How IWAE differs from a standard VAE
-------------------------------------
A VAE optimises a single-sample ELBO:  E_q[log p(x|z) - KL(q||p)].
IWAE tightens this bound by averaging over K independent latent samples
before taking the log.  The resulting estimator is strictly tighter than
the single-sample ELBO for K > 1, and converges to the true log-likelihood
as K → ∞.  The trade-off is K× more forward/backward passes per step.

The resample flag
------------------
Setting self.resample = True tells BaseModel.reparameterize() and
AbsModel.forward() to draw K samples per data point and tile them to
shape (K*batch, z_dim).  IWAE.calculate_loss() then reshapes back to
(K, batch) to apply the logsumexp correctly.  Without resample=True,
forward() would return only one sample and this class would be identical
to a vanilla VAE.

References:
  * https://arxiv.org/abs/1509.00519 (Burda et al., 2015)
  * https://arxiv.org/abs/1705.10306 (Cremer et al., 2017)
"""

from __future__ import print_function
import math
import warnings

import torch

from models.VAE import VAE


class IWAE(VAE):
    """Importance Weighted Autoencoder with K latent samples per step.

    Reuses the VAE encoder/decoder architecture unchanged.  Only the loss
    function differs: instead of a single-sample ELBO, IWAE evaluates K
    importance-weighted samples and applies logsumexp to obtain a tighter
    lower bound on log p(x).

    Args:
        args: Namespace of hyperparameters.  Key IWAE-specific fields:
              K (int) — number of importance samples; should be >= 5 for
              a meaningfully tighter bound.  K=1 reduces to standard VAE.
    """

    def __init__(self, args):
        super(IWAE, self).__init__(args)
        # Enable K-sample resampling in BaseModel.reparameterize() and
        # AbsModel.forward().  This causes both methods to return tensors of
        # shape (K*batch, ...) instead of (batch, ...), which calculate_loss()
        # below then reshapes to (K, batch) for the logsumexp.
        self.resample = True

        if getattr(self.args, 'K', 1) <= 1:
            warnings.warn(
                "IWAE with K=1 is equivalent to a standard VAE. "
                "Consider using --K 5 or higher for a tighter bound.")

    # create_model() is intentionally not overridden.
    # IWAE reuses VAE's encoder/decoder architecture (q_z_layers, p_x_layers,
    # q_z_mean/logvar, p_x_mean) unchanged.  The only difference is in how
    # gradients are computed through the K latent samples, not in which
    # layers are used.  Inheriting create_model() keeps the two classes in sync
    # automatically whenever VAE's architecture changes.

    def calculate_loss(self, x, beta=1., average=False,
                       exemplars_embedding=None, cache=None, dataset=None):
        """Compute the importance-weighted ELBO (IWAE bound) for a batch.

        Unlike the standard ELBO (which averages K terms linearly), IWAE first
        computes per-sample log-weights log w_k = log p(x|z_k) - KL_k and then
        applies logsumexp - log(K) over the K axis.  This is a tighter lower
        bound on log p(x) and produces lower-variance gradient estimates for
        the encoder than a naive K-sample average.

        Args:
            x: Tuple (data, x_indices).
               data      — input batch, shape (batch, D).
               x_indices — training-set indices, shape (batch, 1), used for
                           exemplar prior LOO masking.
            beta:    KL weight for beta-VAE / annealing schedules.
            average: If True, return scalar means over the batch dimension.
            exemplars_embedding: Pre-computed prior parameters, or None.
            cache:   Cached encoder outputs for ANN exemplar search.
            dataset: Full training TensorDataset for on-the-fly exemplar sampling.

        Returns:
            loss: IWAE loss = -(logsumexp(log_w, dim=0) - log K).
                  Shape: scalar if average=True, else (batch,).
            RE:   Mean reconstruction log-likelihood across K samples (monitoring only).
                  Shape: scalar if average=True, else (batch,).
            KL:   Mean KL divergence across K samples (monitoring only).
                  Shape: scalar if average=True, else (batch,).
        """
        x, x_indices = x
        batch_size = x.size(0)
        K = self.args.K

        if self.args.prior == 'exemplar_prior':
            raise NotImplementedError(
                "K > 1 is not supported with exemplar_prior")

        # ----- forward pass -----
        # Because self.resample=True, AbsModel.forward() internally calls
        # reparameterize() with K samples, so all returned tensors already
        # have the K dimension folded in.
        x_mean, x_logvar, latent_stats = self.forward(x)
        # x_mean, x_logvar: (K*batch, D)
        # latent_stats: (z_q, z_q_mean, z_q_logvar), each (K*batch, z_dim)

        # ----- expand x to match the K-tiled decoder outputs -----
        # reconstruction_loss() requires x and x_mean to have identical shapes.
        # x is still (batch, D); we tile it K times along a new leading axis.
        x_expanded = (
            x.unsqueeze(0).expand(K, -1, -1)   # (K, batch, D)
            .contiguous().view(-1, x.size(-1))  # (K*batch, D)
        )

        # ----- per-sample log-probabilities -----
        # log p(x|z) — reconstruction log-likelihood
        log_PxGz = self.reconstruction_loss(x_expanded, x_mean, x_logvar)
        # shape: (K*batch,)

        # KL(q(z|x) || p(z)) = -(log p(z) - log q(z|x)) — positive by convention
        kl = self.kl_loss(latent_stats, exemplars_embedding, dataset, cache, x_indices)
        # shape: (K*batch,)

        # ----- reshape to (K, batch) for the importance-weighted ELBO -----
        log_PxGz = log_PxGz.view(K, batch_size)  # (K, batch)
        kl = kl.view(K, batch_size)               # (K, batch)

        # ----- IWAE bound (Burda et al. 2015, eq. 8) -----
        # log importance weights: log w_k = log p(x|z_k) + log p(z_k) - log q(z_k|x)
        #   = log_PxGz - kl
        # (kl already encodes the sign: kl = -(log p(z) - log q(z|x)), so
        #  log p(x|z) + log p(z) - log q(z|x) = log_PxGz - kl)
        #
        # The IWAE objective is:
        #   L_K = E[log(1/K * sum_k exp(log w_k))]
        #       = E[logsumexp(log_w, dim=0) - log K]
        # We minimise its negative:
        log_w = log_PxGz - beta * kl                               # (K, batch)
        loss = -(torch.logsumexp(log_w, dim=0) - math.log(K))      # (batch,)

        # ----- monitoring metrics (not part of the IWAE gradient) -----
        # RE and KL are K-averaged for logging/tensorboard; the actual
        # optimisation signal comes entirely from `loss` above via logsumexp.
        RE = log_PxGz.mean(dim=0)     # (batch,)
        KL = kl.mean(dim=0)           # (batch,)

        if average:
            loss = loss.mean()
            RE = RE.mean()
            KL = KL.mean()

        return loss, RE, KL
