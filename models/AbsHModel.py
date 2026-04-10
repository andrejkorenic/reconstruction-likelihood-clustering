from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
from utils.distributions import log_normal_diag
from .BaseModel import BaseModel


class BaseHModel(BaseModel):
    """Abstract base for two-level hierarchical VAEs (z1, z2).

    Graphical model:
        z2 ~ p(z2)           # top-level latent (prior depends on prior type)
        z1 ~ p(z1 | z2)      # bottom-level latent (learned conditional)
        x  ~ p(x | z1, z2)   # observation

    Inference model:
        z2 ~ q(z2 | x)       # top-level encoder (reuses BaseModel.q_z)
        z1 ~ q(z1 | x, z2)   # bottom-level encoder (uses both x and z2)

    Subclasses (HVAE_2level, convHVAE_2level, PixelCNN) define the
    network layers via create_model(); this class provides the shared
    forward pass, KL computation, and generation logic.
    """

    def __init__(self, args):
        super(BaseHModel, self).__init__(args)

    # ------------------------------------------------------------------
    # KL divergence: KL(q(z1,z2|x) || p(z1|z2)p(z2))
    # ------------------------------------------------------------------
    def kl_loss(self, latent_stats, exemplars_embedding, dataset, cache, x_indices):
        """Compute KL divergence for both latent levels.

        KL = -[log p(z1|z2) + log p(z2) - log q(z1|x,z2) - log q(z2|x)]

        Args:
            latent_stats: tuple of (z1_q, z1_q_mean, z1_q_logvar,
                          z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar)
            exemplars_embedding: precomputed prior embeddings (or None)
            dataset, cache, x_indices: for exemplar prior LOO masking

        Returns:
            KL divergence, shape (batch,)
        """
        z1_q, z1_q_mean, z1_q_logvar, z2_q, z2_q_mean, z2_q_logvar, z1_p_mean, z1_p_logvar = latent_stats
        if exemplars_embedding is None and self.args.prior == 'exemplar_prior':
            exemplars_embedding = self.get_exemplar_set(z2_q_mean, z2_q_logvar,
                                                        dataset, cache, x_indices)

        # z1: conditional prior p(z1|z2) vs q(z1|x,z2)
        log_p_z1 = log_normal_diag(z1_q.view(-1, self.args.z1_size),
                                   z1_p_mean.view(-1, self.args.z1_size),
                                   z1_p_logvar.view(-1, self.args.z1_size), dim=1)
        log_q_z1 = log_normal_diag(z1_q.view(-1, self.args.z1_size),
                                   z1_q_mean.view(-1, self.args.z1_size),
                                   z1_q_logvar.view(-1, self.args.z1_size), dim=1)

        # z2: marginal prior p(z2) vs q(z2|x)
        log_p_z2 = self.log_p_z(z=(z2_q, x_indices),
                                exemplars_embedding=exemplars_embedding)
        log_q_z2 = log_normal_diag(z2_q.view(-1, self.args.z2_size),
                                   z2_q_mean.view(-1, self.args.z2_size),
                                   z2_q_logvar.view(-1, self.args.z2_size), dim=1)

        return -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)

    # ------------------------------------------------------------------
    # Generation: z2 → z1 → x
    # ------------------------------------------------------------------
    def generate_x_from_z(self, z, with_reparameterize=True):
        """Generate observations from top-level latent z2.

        z2 → p(z1|z2) → sample z1 → p(x|z1,z2) → x_mean
        """
        z1_sample_mean, z1_sample_logvar = self.p_z1(z)
        if with_reparameterize:
            z1_sample_rand = self.reparameterize(z1_sample_mean, z1_sample_logvar)
        else:
            z1_sample_rand = z1_sample_mean

        if self.args.model_name == 'pixelcnn':
            generated_xs = self.pixelcnn_generate(
                z1_sample_rand.view(-1, self.args.z1_size),
                z.reshape(-1, self.args.z2_size))
        else:
            generated_xs, _ = self.p_x(
                z1_sample_rand.view(-1, self.args.z1_size),
                z.view(-1, self.args.z2_size))
        return generated_xs

    # ------------------------------------------------------------------
    # Conditional prior: p(z1 | z2)
    # ------------------------------------------------------------------
    def p_z1(self, z2):
        """Compute p(z1 | z2) parameters. Subclasses define p_z1_layers_z2."""
        z2 = self.p_z1_layers_z2(z2)
        z1_p_mean = self.p_z1_mean(z2)
        z1_p_logvar = self.p_z1_logvar(z2)
        return z1_p_mean, z1_p_logvar

    # ------------------------------------------------------------------
    # Bottom-level encoder: q(z1 | x, z2)
    # ------------------------------------------------------------------
    def q_z1(self, x, z2):
        """Encode x and z2 jointly to produce q(z1 | x, z2) parameters."""
        x = self.q_z1_layers_x(x)
        if self.args.model_name == 'convhvae_2level' or self.args.model_name == 'pixelcnn':
            x = x.view(x.size(0), -1)  # flatten conv features
        z2 = self.q_z1_layers_z2(z2)
        h = torch.cat((x, z2), 1)
        h = self.q_z1_layers_joint(h)
        z1_q_mean = self.q_z1_mean(h)
        z1_q_logvar = self.q_z1_logvar(h)
        return z1_q_mean, z1_q_logvar

    # ------------------------------------------------------------------
    # Decoder: p(x | z1, z2)
    # ------------------------------------------------------------------
    def p_x(self, z1, z2, x=None):
        """Decode z1 and z2 to observation distribution parameters.

        Args:
            z1: bottom-level latent, shape (batch, z1_size)
            z2: top-level latent, shape (batch, z2_size)
            x:  original input (only used by PixelCNN for autoregressive conditioning)

        Returns:
            (x_mean, x_logvar) — reconstruction distribution parameters
        """
        z1 = self.p_x_layers_z1(z1)
        z2 = self.p_x_layers_z2(z2)

        if self.args.model_name == 'pixelcnn':
            # PixelCNN: autoregressive decoder conditioned on (x, z1, z2)
            z2 = z2.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
            z1 = z1.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
            h = torch.cat((x, z1, z2), 1)
            h_decoder = self.pixelcnn(h)
        else:
            # Standard decoder: MLP or conv on concatenated (z1, z2)
            h = torch.cat((z1, z2), 1)
            if 'convhvae_2level' in self.args.model_name:
                h = self.p_x_layers_joint_pre(h)
                # Reshape to spatial bottleneck (64, H//4, W//4) for transposed conv upsampling
                h = h.view(-1, 64, self.args.input_size[1] // 4, self.args.input_size[2] // 4)
            h_decoder = self.p_x_layers_joint(h)

        x_mean = self.p_x_mean(h_decoder)
        if 'convhvae_2level' in self.args.model_name or self.args.model_name == 'pixelcnn':
            x_mean = x_mean.view(-1, np.prod(self.args.input_size))

        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            if getattr(self.args, 'use_logit', False) is False:
                x_mean = torch.clamp(x_mean, min=0. + 1./512., max=1. - 1./512.)
                x_logvar = self.decoder_logstd * x_mean.new_ones(size=x_mean.shape)
            else:
                x_logvar = self.p_x_logvar(h_decoder)
                if 'convhvae_2level' in self.args.model_name or self.args.model_name == 'pixelcnn':
                    x_logvar = x_logvar.view(-1, np.prod(self.args.input_size))

        return x_mean, x_logvar

    # ------------------------------------------------------------------
    # Forward pass: full encode → decode pipeline
    # ------------------------------------------------------------------
    def forward(self, x):
        """Full forward pass: x → q(z2|x) → q(z1|x,z2) → p(x|z1,z2).

        Returns:
            (x_mean, x_logvar, latent_stats) where latent_stats is an 8-tuple
            passed to kl_loss().
        """
        if self.resample:
            raise NotImplementedError(
                "K > 1 is not supported for hierarchical models")

        # Top-level: q(z2 | x)
        z2_q_mean, z2_q_logvar = self.q_z(x)
        z2_q = self.reparameterize(z2_q_mean, z2_q_logvar)

        # Bottom-level: q(z1 | x, z2)
        z1_q_mean, z1_q_logvar = self.q_z1(x, z2_q)
        z1_q = self.reparameterize(z1_q_mean, z1_q_logvar)

        # Conditional prior: p(z1 | z2)
        z1_p_mean, z1_p_logvar = self.p_z1(z2_q)

        # Decoder: p(x | z1, z2)
        if self.args.model_name == 'pixelcnn':
            x_mean, x_logvar = self.p_x(z1_q, z2_q, x=x)
        else:
            x_mean, x_logvar = self.p_x(z1_q, z2_q)

        return x_mean, x_logvar, (z1_q, z1_q_mean, z1_q_logvar,
                                  z2_q, z2_q_mean, z2_q_logvar,
                                  z1_p_mean, z1_p_logvar)

