"""ConvVAE — single-level convolutional Variational Autoencoder.

Implements a standard VAE with a gated-convolutional encoder and a
transposed-convolution decoder.  Inherits the shared training
infrastructure via BaseModel -> AbsModel -> ConvVAE:

    BaseModel  — prior logic, calculate_loss, reparameterize, q_z, generate_z
    AbsModel   — kl_loss, forward, p_x, generate_x_from_z  (single-level specific)
    ConvVAE    — create_model() wires up the convolutional layer definitions

Encoder:
    x -> 4x GatedConv2d (two with stride-2 for spatial downsampling)
    -> flatten -> (z_mean, z_logvar)

Decoder:
    z -> GatedDense -> reshape to spatial
    -> 2x GatedConvTranspose2d (stride-2 upsampling)
    -> 1x GatedConv2d (refinement)
    -> Conv2d output head

The flattened bottleneck size h_size = 64 * (H//4) * (W//4) is computed
dynamically from the input spatial dimensions, so the model works with
any dataset whose H and W are divisible by 4 (e.g. MNIST 28x28,
CIFAR-10 32x32, Frey Faces 28x20).

Supported priors (via args.prior): standard, vampprior, exemplar_prior.
Supported likelihoods (via args.input_type): binary (Bernoulli),
gray / continuous (diagonal Normal with per-pixel log-variance).
"""

from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from utils.nn import GatedConv2d, GatedConvTranspose2d, GatedDense, NonLinear, Conv2d
from models.AbsModel import AbsModel


class ConvVAE(AbsModel):
    """Single-level convolutional VAE with gated conv encoder and transposed conv decoder.

    Architecture summary:
        Encoder:  x -> GatedConv2d(7x7) -> GatedConv2d(3x3,s2) -> GatedConv2d(5x5)
                    -> GatedConv2d(3x3,s2) -> flatten -> (z_mean, z_logvar)
        Decoder:  z -> GatedDense -> reshape -> GatedConvTranspose2d(4x4,s2) x2
                    -> GatedConv2d(5x5) refinement -> Conv2d output head

    The bottleneck size h_size = 64 * (H//4) * (W//4) adapts to the
    input resolution automatically.

    Args:
        args: Namespace of hyperparameters (see BaseModel for the full list).
    """

    def __init__(self, args):
        super(ConvVAE, self).__init__(args)

    def create_model(self, args, train_data_size=None):
        """Define all trainable layers for the convolutional encoder and decoder.

        Called by BaseModel.__init__ after the shared output heads have been
        registered.  The Conv2d output heads defined here replace the Linear
        ones from BaseModel.

        Args:
            args: Namespace with hyperparameters.  Key fields:
                  input_size (list[int]) — [C, H, W];
                  z1_size (int)          — latent dimensionality;
                  no_attention (bool)    — disable gating in GatedConv2d layers;
                  input_type (str)       — 'binary', 'gray', or 'continuous'.
            train_data_size: Stored for bookkeeping (e.g. exemplar prior).
        """
        self.train_data_size = train_data_size
        C, H, W = args.input_size[0], args.input_size[1], args.input_size[2]
        self.h_size = 64 * (H // 4) * (W // 4)  # flattened bottleneck size

        # ----- encoder: q(z|x) -----
        # Four GatedConv2d layers; the 2nd and 4th use stride=2 to downsample
        # spatial resolution by 4x total (H -> H/4, W -> W/4).
        self.q_z_layers = nn.Sequential(
            GatedConv2d(C,  32, 7, 1, 3, no_attention=args.no_attention),
            GatedConv2d(32, 32, 3, 2, 1, no_attention=args.no_attention),   # downsample x2
            GatedConv2d(32, 64, 5, 1, 2, no_attention=args.no_attention),
            GatedConv2d(64, 64, 3, 2, 1, no_attention=args.no_attention),   # downsample x2
        )
        # Posterior mean: unconstrained linear projection.
        self.q_z_mean = NonLinear(self.h_size, args.z1_size, activation=None)
        # Posterior log-variance: clamped to [-6, 2] to avoid collapse or blowup.
        self.q_z_logvar = NonLinear(self.h_size, args.z1_size,
                                    activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ----- decoder: p(x|z) -----
        # GatedDense projects z back to the flattened spatial bottleneck,
        # then two GatedConvTranspose2d layers upsample back to full resolution,
        # followed by a refinement GatedConv2d.
        self.p_x_layer_z = GatedDense(args.z1_size, self.h_size,
                                       no_attention=args.no_attention)
        self.p_x_layers = nn.Sequential(
            GatedConvTranspose2d(64, 64, 4, 2, 1, no_attention=args.no_attention),  # upsample x2
            GatedConvTranspose2d(64, 32, 4, 2, 1, no_attention=args.no_attention),  # upsample x2
            GatedConv2d(32, 32, 5, 1, 2, no_attention=args.no_attention),           # refinement
        )

        # ----- output head (replaces the Linear p_x_mean/p_x_logvar from BaseModel.__init__) -----
        if args.input_type == 'binary':
            self.p_x_mean = Conv2d(32, C, 1, 1, 0, activation=nn.Sigmoid())
        elif args.input_type in ('gray', 'continuous'):
            self.p_x_mean = Conv2d(32, C, 1, 1, 0)
            self.p_x_logvar = Conv2d(32, C, 1, 1, 0,
                                      activation=nn.Hardtanh(min_val=-4.5, max_val=0.))

    def q_z(self, x, prior=False):
        """Encode input into posterior parameters.

        Args:
            x: Input tensor, shape (B, C, H, W) or flat (B, C*H*W).
               Flat input is automatically reshaped to spatial.
            prior: If True and using exemplar_prior, replace the learned
                   per-input log-variance with the shared prior_log_variance
                   scalar (broadcast to (B, z1_size)).  This is used by
                   generate_z() and get_exemplar_set() to sample from the
                   prior's encoder distribution rather than the posterior.

        Returns:
            z_mean:   Posterior mean, shape (B, z1_size).
            z_logvar: Posterior log-variance, shape (B, z1_size).
        """
        # auto-reshape flat input to spatial (matches forward() behavior)
        if x.dim() == 2:
            C, H, W = self.args.input_size
            x = x.view(-1, C, H, W)
        h = self.q_z_layers(x)          # (B, 64, H/4, W/4)
        h = h.view(h.size(0), -1)       # (B, h_size)
        z_mean = self.q_z_mean(h)       # (B, z1_size)
        if prior is True and self.args.prior == 'exemplar_prior':
            z_logvar = self.prior_log_variance * torch.ones(
                (x.shape[0], self.args.z1_size), device=x.device)
        else:
            z_logvar = self.q_z_logvar(h)   # (B, z1_size)
        return z_mean, z_logvar

    def p_x(self, z):
        """Decode latent vector into reconstruction parameters.

        Overrides AbsModel.p_x to use the transposed-conv decoder stack
        with spatial feature maps instead of fully-connected layers.

        Args:
            z: Latent samples, shape (B, z1_size) or (B, N, z1_size).
               Extra batch dims (from generation) are flattened internally.

        Returns:
            x_mean:   Reconstructed pixel means, shape (B, prod(input_size)).
            x_logvar: Per-pixel log-variance, shape (B, prod(input_size)).
                      Zeros for binary input_type.
                      For gray/continuous without logit: global decoder_logstd.
                      For gray/continuous with logit: per-pixel from p_x_logvar conv head.
        """
        C, H, W = self.args.input_size
        D = np.prod(self.args.input_size)

        # flatten extra batch dims (e.g. generation passes (B, N, z1_size))
        z = z.reshape(-1, z.shape[-1])                    # (B*, z1_size)

        h = self.p_x_layer_z(z)                           # (B*, h_size)
        h = h.view(h.size(0), 64, H // 4, W // 4)        # (B*, 64, H/4, W/4)
        h = self.p_x_layers(h)                            # (B*, 32, H, W)
        x_mean = self.p_x_mean(h)                         # (B*, C, H, W)
        x_mean = x_mean.view(-1, D)                       # (B*, C*H*W)

        if self.args.input_type == 'binary':
            x_logvar = torch.zeros(1, D, device=x_mean.device)
        else:
            if self.args.use_logit is False:
                # Clamp x_mean strictly inside (0, 1) for log_logistic_256
                x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
                # Global learned log-std broadcast to match x_mean shape
                x_logvar = self.decoder_logstd * x_mean.new_ones(size=x_mean.shape)
            else:
                # use_logit=True: data in unbounded logit space, use per-pixel log-variance
                x_logvar_spatial = self.p_x_logvar(h)     # (B*, C, H, W)
                x_logvar = x_logvar_spatial.view(-1, D)

        return x_mean, x_logvar

    def forward(self, x, label=0, num_categories=10):
        """Reshape flat input to spatial, then run standard VAE pipeline.

        Args:
            x: Input batch, shape (B, C*H*W) — flat pixel vector.

        Returns:
            See AbsModel.forward for return format.
        """
        C, H, W = self.args.input_size
        x = x.view(-1, C, H, W)          # reshape flat -> spatial
        return super().forward(x)          # AbsModel.forward handles the rest
