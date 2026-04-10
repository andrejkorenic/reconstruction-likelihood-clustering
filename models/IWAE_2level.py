"""
Importance Weighted Autoencoder with two stochastic hidden layers.

Extends the two-level hierarchical VAE (HVAE_2level) with the importance-weighted
ELBO from Burda et al. (2015). Uses our existing top-down inference architecture:
q(z2|x) * q(z1|x,z2) with decoder p(x|z1,z2).

The IWAE bound for two stochastic layers:

    L_K(x) = E[log(1/K * sum_k w_k)]

where the importance weights decompose as:

    log w_k = log p(x|z1_k,z2_k) + log p(z1_k|z2_k) + log p(z2_k)
              - log q(z2_k|x) - log q(z1_k|x,z2_k)

This simplifies to log w_k = log p(x|z) - KL, using the same decomposition
as AbsHModel.kl_loss().

How IWAE_2level differs from HVAE_2level
-----------------------------------------
HVAE_2level.forward() raises NotImplementedError when resample=True.
IWAE_2level overrides forward() with a custom K-sampling strategy that is
specific to the two-level hierarchy:
  - z2 is sampled K times per data point (the outer, top-level stochastic layer).
  - z1 is sampled once per (x, z2) pair (one sample per "path").
This is more principled than simply enabling resample=True on the base class,
because it preserves the conditional dependency z1 ~ q(z1|x, z2) correctly
for each of the K z2 samples independently.

The resample=False default
---------------------------
self.resample stays False (the BaseModel default) in __init__.  The K-sampling
logic is handled entirely inside the overridden forward(), not delegated to
BaseModel.reparameterize().  Setting resample=True here would cause
BaseHModel.forward() to raise NotImplementedError before IWAE_2level.forward()
could run, and would also double-apply the tiling logic.

References:
  * Burda, Y., Grosse, R., & Salakhutdinov, R. (2015).
    Importance Weighted Autoencoders. https://arxiv.org/abs/1509.00519
  * Cremer, C., Morris, Q., & Duvenaud, D. (2017).
    Reinterpreting Importance-Weighted Autoencoders.
    https://arxiv.org/abs/1705.10306
  * Rainforth, T., Kosiorek, A., Le, T. A., Maddison, C., Igl, M.,
    Wood, F., & Teh, Y. W. (2018). Tighter Variational Bounds are Not
    Necessarily Better. https://arxiv.org/abs/1802.04537
"""

from __future__ import print_function
import math
import warnings

import torch

from models.HVAE_2level import VAE as HVAE_2level


class IWAE_2level(HVAE_2level):
    """Two-level IWAE: hierarchical VAE trained with K importance samples.

    Architecture (inherited from HVAE_2level):
        Encoder:  x → q(z2|x) → q(z1|x, z2)
        Prior:    p(z2) [configured via args.prior], p(z1|z2)
        Decoder:  p(x|z1, z2)

    The importance-weighted ELBO is computed by:
      1. Drawing K z2 samples from q(z2|x).
      2. Drawing one z1 sample per (x, z2) pair from q(z1|x, z2).
      3. Computing log w_k = log p(x|z1,z2) - KL for each of the K paths.
      4. Returning -(logsumexp(log_w) - log K) as the loss.

    Args:
        args: Namespace with hyperparameters.  Key fields:
              K (int)      — number of importance samples (>= 5 recommended).
              z1_size      — dimensionality of z1 (fine-grained latent).
              z2_size      — dimensionality of z2 (coarse-grained latent).
    """

    def __init__(self, args):
        super(IWAE_2level, self).__init__(args)
        # resample stays False — we handle K-sampling manually in forward()
        # because z2 needs K samples but z1 needs only 1 per z2

        if getattr(self.args, 'K', 1) <= 1:
            warnings.warn(
                "IWAE with K=1 is equivalent to a standard HVAE. "
                "Consider using --K 5 or higher for a tighter bound.")

    def forward(self, x):
        """Forward pass with K importance samples for two stochastic layers.

        Sampling strategy:
          - z2: K samples per data point via Normal.rsample([K])
          - z1: 1 sample per (x, z2) pair via Normal.rsample()

        Why not use self.resample / BaseModel.reparameterize()?
        The base reparameterize() draws K samples of a single variable.  Here
        we need to tile x before encoding z1, because q(z1|x, z2) is conditioned
        on both x AND a specific z2 sample.  Doing this manually avoids the
        NotImplementedError in BaseHModel.forward() and ensures that each z1_k
        is properly conditioned on its own z2_k.

        All outputs have shape (K*batch, ...) for consistent downstream
        processing by kl_loss() and calculate_loss().

        Args:
            x: Input batch, shape (batch, D).

        Returns:
            x_mean:      Decoder pixel means, shape (K*batch, D).
            x_logvar:    Decoder log-variance, shape (K*batch, D).
            latent_stats: 8-tuple matching the format expected by
                          BaseHModel.kl_loss():
                          (z1_q, z1_q_mean, z1_q_logvar,
                           z2_q, z2_q_mean, z2_q_logvar,
                           z1_p_mean, z1_p_logvar)
                          All tensors have shape (K*batch, respective_dim).
        """
        K = self.args.K
        batch_size = x.size(0)

        # ----- step 1: encode z2 from x -----
        z2_q_mean, z2_q_logvar = self.q_z(x)  # each: (batch, z2_dim)

        # ----- step 2: draw K z2 samples per data point -----
        # We call rsample([K]) directly rather than using self.reparameterize()
        # to avoid the resample flag / NotImplementedError path in BaseHModel.
        z2_std = torch.exp(0.5 * z2_q_logvar)
        z2_dist = torch.distributions.Normal(z2_q_mean, z2_std)
        z2_q = z2_dist.rsample([K])                      # (K, batch, z2_dim)
        z2_q = z2_q.view(K * batch_size, -1)              # (K*batch, z2_dim)

        # ----- step 3: expand x to match z2's K-tiled batch -----
        # q(z1|x, z2) needs x in (K*batch, D) form so that each z2_k sample
        # is paired with a copy of the corresponding original x.
        x_expanded = (
            x.unsqueeze(0).expand(K, -1, -1)   # (K, batch, D)
            .contiguous().view(-1, x.size(-1))  # (K*batch, D)
        )

        # ----- step 4: encode z1 from (x, z2) -----
        z1_q_mean, z1_q_logvar = self.q_z1(x_expanded, z2_q)  # each: (K*batch, z1_dim)

        # ----- step 5: draw ONE z1 sample per (x, z2) pair -----
        # A single z1 per path is standard for the hierarchical IWAE.
        # Drawing multiple z1 per z2 would require an additional logsumexp
        # dimension and is not commonly done.
        z1_std = torch.exp(0.5 * z1_q_logvar)
        z1_dist = torch.distributions.Normal(z1_q_mean, z1_std)
        z1_q = z1_dist.rsample()                          # (K*batch, z1_dim)

        # ----- step 6: prior p(z1|z2) and decoder p(x|z1,z2) -----
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)          # each: (K*batch, z1_dim)
        x_mean, x_logvar = self.p_x(z1_q, z2_q)           # each: (K*batch, D)

        # ----- step 7: expand z2 posterior params for kl_loss() -----
        # BaseHModel.kl_loss() expects z2_q_mean and z2_q_logvar to have the
        # same leading dimension as z2_q (K*batch), so it can compute
        # log q(z2|x) per sample.  The original tensors are (batch, z2_dim);
        # we tile them K times to align with the K z2 samples drawn in step 2.
        z2_q_mean = (
            z2_q_mean.unsqueeze(0).expand(K, -1, -1)       # (K, batch, z2_dim)
            .contiguous().view(-1, self.args.z2_size)       # (K*batch, z2_dim)
        )
        z2_q_logvar = (
            z2_q_logvar.unsqueeze(0).expand(K, -1, -1)     # (K, batch, z2_dim)
            .contiguous().view(-1, self.args.z2_size)       # (K*batch, z2_dim)
        )

        return x_mean, x_logvar, (
            z1_q, z1_q_mean, z1_q_logvar,
            z2_q, z2_q_mean, z2_q_logvar,
            z1_p_mean, z1_p_logvar
        )

    def calculate_loss(self, x, beta=1., average=False,
                       exemplars_embedding=None, cache=None, dataset=None):
        """Compute the importance-weighted ELBO for the two-level hierarchy.

        The IWAE bound for two stochastic layers factors as:
            log w_k = log p(x|z1_k,z2_k)
                    + log p(z1_k|z2_k) + log p(z2_k)
                    - log q(z1_k|x,z2_k) - log q(z2_k|x)

        BaseHModel.kl_loss() already computes the combined KL for both layers:
            KL = -(log p(z1|z2) + log p(z2) - log q(z1|x,z2) - log q(z2|x))

        so log w_k = log_PxGz - kl, exactly as in the single-layer IWAE.

        Args:
            x: Tuple (data, x_indices).
               data      — input batch, shape (batch, D).
               x_indices — training-set indices, shape (batch, 1).
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
        # All returned tensors have shape (K*batch, ...) — see forward() above.
        x_mean, x_logvar, latent_stats = self.forward(x)

        # ----- expand x to match the K-tiled decoder outputs -----
        x_expanded = (
            x.unsqueeze(0).expand(K, -1, -1)   # (K, batch, D)
            .contiguous().view(-1, x.size(-1))  # (K*batch, D)
        )

        # ----- per-sample log-probabilities -----
        # log p(x|z1,z2) — reconstruction log-likelihood
        log_PxGz = self.reconstruction_loss(x_expanded, x_mean, x_logvar)
        # shape: (K*batch,)

        # KL(q(z1,z2|x) || p(z1,z2)) — combined KL over both stochastic layers.
        # BaseHModel.kl_loss() computes:
        #   -(log p(z1|z2) + log p(z2) - log q(z1|x,z2) - log q(z2|x))
        kl = self.kl_loss(latent_stats, exemplars_embedding, dataset, cache,
                          x_indices)
        # shape: (K*batch,)

        # ----- reshape to (K, batch) for the importance-weighted ELBO -----
        log_PxGz = log_PxGz.view(K, batch_size)  # (K, batch)
        kl = kl.view(K, batch_size)               # (K, batch)

        # ----- IWAE bound (Burda et al. 2015, eq. 8) -----
        # log w_k = log p(x|z1,z2) + log p(z1|z2) + log p(z2)
        #           - log q(z2|x) - log q(z1|x,z2)
        #         = log_PxGz - kl  (same decomposition as single-layer IWAE)
        log_w = log_PxGz - beta * kl                           # (K, batch)
        loss = -(torch.logsumexp(log_w, dim=0) - math.log(K))  # (batch,)

        # ----- monitoring metrics (not part of the IWAE gradient) -----
        # RE and KL are K-averaged for logging; the actual optimisation signal
        # comes entirely from `loss` above via logsumexp.
        RE = log_PxGz.mean(dim=0)     # (batch,)
        KL = kl.mean(dim=0)           # (batch,)

        if average:
            loss = loss.mean()
            RE = RE.mean()
            KL = KL.mean()

        return loss, RE, KL
