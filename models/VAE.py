"""VAE — flat (single-level) Variational Autoencoder.

Implements the standard VAE (Kingma & Welling, 2013) with a fully-connected
encoder and decoder.  The model plugs into the shared training infrastructure
via the BaseModel → AbsModel inheritance chain:

    BaseModel  — prior logic, calculate_loss, reparameterize, q_z, generate_z
    AbsModel   — kl_loss, forward, p_x, generate_x_from_z  (single-level specific)
    VAE        — create_model() wires up the concrete layer definitions

Supported priors (configured via args.prior): standard, vampprior, exemplar_prior.
Supported likelihoods (configured via args.input_type): binary (Bernoulli),
gray / continuous (logistic-256 or diagonal Normal).
"""

from __future__ import print_function
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from utils.nn import GatedDense, NonLinear
from models.AbsModel import AbsModel


class VAE(AbsModel):
    """Flat VAE with a two-layer gated-dense encoder and decoder.

    Architecture summary:
        Encoder:  x → GatedDense → GatedDense → (z_mean, z_logvar)
        Decoder:  z → GatedDense → GatedDense → (x_mean [, x_logvar])

    Both encoder and decoder use two hidden layers of size args.hidden_size.
    The output heads (p_x_mean, p_x_logvar) are defined in BaseModel.__init__
    because they depend on args.input_type and are shared across all AbsModel
    subclasses.

    Args:
        args: Namespace of hyperparameters (see BaseModel for the full list).
    """

    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args, train_data_size=None):
        """Define all trainable layers for the VAE encoder and decoder.

        Called by BaseModel.__init__ after the shared output heads (p_x_mean,
        p_x_logvar, decoder_logstd) have already been registered, so those do
        not need to be created here.

        Args:
            args: Namespace with hyperparameters.  Key fields used here:
                  input_size (list[int]) — spatial dims, flattened for the FC input;
                  hidden_size (int)      — width of each hidden layer;
                  z1_size (int)          — dimensionality of the latent space;
                  no_attention (bool)    — disable the gating branch in GatedDense;
                  same_variational_var (bool) — share a single scalar log-variance
                                               across all latent dimensions.
            train_data_size: Stored but not used at construction time; available
                             later for bookkeeping (e.g. exemplar prior sampling).
        """
        self.train_data_size = train_data_size

        # ----- encoder: q(z|x) -----
        # Two stacked GatedDense layers map the flattened input to a hidden
        # representation that is then split into the posterior mean and log-variance.
        #
        # GatedDense computes  h(x) * σ(g(x))  where h and g are separate linear
        # projections.  The element-wise sigmoid gate acts as a soft feature
        # selector, giving the encoder capacity to suppress irrelevant input
        # dimensions without requiring explicit feature engineering.
        # When no_attention=True the gate is replaced by a plain ReLU (cheaper,
        # but typically worse at capturing fine-grained structure).
        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), self.args.hidden_size, no_attention=self.args.no_attention),
            GatedDense(self.args.hidden_size, self.args.hidden_size, no_attention=self.args.no_attention)
        )
        # Posterior mean: unconstrained linear projection — the reparameterization
        # trick handles the stochasticity, so no activation is needed here.
        self.q_z_mean = Linear(self.args.hidden_size, self.args.z1_size)   # (batch, z1_size)

        if args.same_variational_var:
            # A single global log-variance shared by all z dimensions and all data
            # points.  Useful as a diagnostic ablation: if the model can learn with
            # a fixed isotropic posterior variance it suggests the prior is well-matched.
            self.q_z_logvar = torch.nn.Parameter(torch.randn((1)))
        else:
            # Per-dimension, per-datapoint log-variance.  Hardtanh clamps to [-6, 2]
            # to prevent (a) posterior collapse (logvar → -∞, zero KL) and
            # (b) runaway variance (logvar → +∞, encoder ignores the data).
            # The asymmetric range [-6, 2] allows tight posteriors while keeping
            # the maximum std at exp(1) ≈ 2.7, well within a reasonable range.
            self.q_z_logvar = NonLinear(self.args.hidden_size,
                                        self.args.z1_size, activation=nn.Hardtanh(min_val=-6., max_val=2.))
            # output shape: (batch, z1_size)

        # ----- decoder: p(x|z) -----
        # Mirrors the encoder topology: two GatedDense layers map z back to a
        # hidden representation of the same width.  The output heads that project
        # from hidden_size to pixel space (p_x_mean, p_x_logvar / decoder_logstd)
        # are defined in BaseModel.__init__ and called in AbsModel.p_x().
        self.p_x_layers = nn.Sequential(
            GatedDense(self.args.z1_size, self.args.hidden_size, no_attention=self.args.no_attention),
            GatedDense(self.args.hidden_size, self.args.hidden_size, no_attention=self.args.no_attention))
        # output shape after p_x_layers: (batch, hidden_size)
