from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from models.BaseModel import BaseModel
from utils.distributions import log_normal_diag


class AbsModel(BaseModel):
    def __init__(self, args):
        super(AbsModel, self).__init__(args)

    def kl_loss(self, latent_stats, exemplars_embedding, dataset, cache, x_indices):
        """Compute KL(q(z|x) || p(z)) = -(log p(z) - log q(z|x))."""
        z_q, z_q_mean, z_q_logvar = latent_stats
        if exemplars_embedding is None and self.args.prior == 'exemplar_prior':
            exemplars_embedding = self.get_exemplar_set(z_q_mean, z_q_logvar, dataset, cache, x_indices)
        log_p_z = self.log_p_z(z=(z_q, x_indices), exemplars_embedding=exemplars_embedding)  # log p(z)
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)  # log q(z|x)
        return -(log_p_z - log_q_z)  # KL(q(z|x) || p(z)), positive by convention

    def generate_x_from_z(self, z, with_reparameterize=True):
        """Decode a latent vector z into pixel space.

        Args:
            z: Latent samples, shape (N, z1_size).
            with_reparameterize: Unused in this implementation (kept for API
                compatibility with callers that pass it explicitly).

        Returns:
            generated_x: Reconstructed pixel values, shape (N, prod(input_size)).
                         If use_logit=True, values are in [0, 1] after the logit
                         inverse transform; otherwise the raw decoder output is
                         returned (sigmoid-squashed for binary, unclamped for
                         continuous).
        """
        generated_x, _ = self.p_x(z)
        try:
            if self.args.use_logit is True:
                # The logit pre-processing transform squashes pixel values from
                # [0, 1] into an unbounded range before passing them to the encoder,
                # which stabilises training on continuous data.  Here we invert it
                # so that generated samples are back in the original [0, 1] space.
                return self.logit_inverse(generated_x)
            else:
                return generated_x
        except:
            # args.use_logit may not exist for older checkpoints — fall back to
            # returning the raw decoder output rather than crashing.
            return generated_x

    def p_x(self, z):
        """Run the decoder to produce reconstruction parameters (x_mean, x_logvar).

        Args:
            z: Latent samples, shape (batch, z1_size).  When K > 1 (IWAE), the
               batch dimension is K*batch_size.

        Returns:
            x_mean:   Reconstructed pixel means, shape (batch, prod(input_size)).
                      For binary inputs: values in (0, 1) from the Sigmoid output head.
                      For continuous inputs without logit: values clamped to
                      (1/512, 1 - 1/512) so they stay strictly inside (0, 1) — this
                      avoids log(0) when evaluating logistic-256 likelihood.
            x_logvar: Per-pixel log-variance, shape (batch, prod(input_size)).
                      For binary inputs: zeros (Bernoulli has no variance parameter).
                      For continuous inputs without logit: a constant tensor filled with
                      decoder_logstd (a learned global scalar) — the logistic-256
                      likelihood uses a single temperature rather than per-pixel variance.
                      For continuous inputs with logit: per-pixel log-variance from
                      p_x_logvar head (defined in BaseModel).
        """
        z = self.p_x_layers(z)           # (batch, hidden_size)
        x_mean = self.p_x_mean(z)        # (batch, prod(input_size))  — activation depends on input_type

        # ----- likelihood-specific variance handling -----
        if self.args.input_type == 'binary':
            # Bernoulli likelihood: x_logvar is unused by log_bernoulli but must
            # be returned to keep the (x_mean, x_logvar) API consistent.
            x_logvar = torch.zeros(1, np.prod(self.args.input_size))
        else:
            if self.args.use_logit is False:
                # Clamp x_mean strictly inside (0, 1) so that log_logistic_256
                # never receives boundary values (log(0) = -inf).
                # 1/512 ≈ 0.002 provides a small but safe margin from 0 and 1.
                x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
                # Use the global learned log-std as a uniform temperature across
                # all pixels.  Broadcast scalar to match x_mean's full shape.
                x_logvar = self.decoder_logstd*x_mean.new_ones(size=x_mean.shape)  # (batch, prod(input_size))
            else:
                # use_logit=True: data in logit space, per-pixel log-variance from head
                x_logvar = self.p_x_logvar(z)  # (batch, prod(input_size))

        return x_mean.reshape(-1, np.prod(self.args.input_size)), x_logvar.reshape(-1, np.prod(self.args.input_size))

    def forward(self, x, label=0, num_categories=10):
        """Run the full VAE encode → reparameterize → decode pipeline.

        Args:
            x:               Input batch, shape (batch, prod(input_size)).
            label:           Unused; kept for API compatibility with conditional models.
            num_categories:  Unused; kept for API compatibility with conditional models.

        Returns:
            x_mean:      Decoder pixel means, shape (batch, prod(input_size)).
                         Shape is (K*batch, prod(input_size)) when resample=True.
            x_logvar:    Decoder log-variance, same shape as x_mean.
            latent_stats: Tuple (z_q, z_q_mean, z_q_logvar):
                          - z_q:        Sampled latent, (batch, z1_size) or (K*batch, z1_size).
                          - z_q_mean:   Posterior mean, same shape as z_q.
                          - z_q_logvar: Posterior log-variance, same shape as z_q.
                          When resample=True, z_q_mean and z_q_logvar are tiled K times
                          to align with the K samples drawn by reparameterize().
        """
        # ----- encode: x → posterior parameters -----
        z_q_mean, z_q_logvar = self.q_z(x)       # each: (batch, z1_size)

        # ----- reparameterize: sample z ~ q(z|x) -----
        # When resample=False (standard VAE): draws one sample → (batch, z1_size).
        # When resample=True  (K-sample IWAE): draws K samples → (K*batch, z1_size).
        z_q = self.reparameterize(z_q_mean, z_q_logvar)  # (batch, z1_size) or (K*batch, z1_size)

        # ----- decode: z → reconstruction parameters -----
        x_mean, x_logvar = self.p_x(z_q)         # each: (batch, D) or (K*batch, D)

        if self.resample:
            # expand posterior params to match z_q's (K*batch, z_dim) shape
            # so that kl_loss() can compute per-sample KL terms correctly.
            K = self.args.K
            z_q_mean = (
                z_q_mean.unsqueeze(0).expand(K, -1, -1)    # (K, batch, z1_size)
                .contiguous().view(-1, self.args.z1_size)   # (K*batch, z1_size)
            )
            z_q_logvar = (
                z_q_logvar.unsqueeze(0).expand(K, -1, -1)  # (K, batch, z1_size)
                .contiguous().view(-1, self.args.z1_size)   # (K*batch, z1_size)
            )

        return x_mean, x_logvar, (z_q, z_q_mean, z_q_logvar)
