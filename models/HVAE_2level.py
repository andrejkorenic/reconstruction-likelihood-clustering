"""
HVAE 2-level — Two-level hierarchical VAE with dense (non-convolutional) layers.

Architecture:
  Encoder:  q(z2|x)       — GatedDense stack, produces z2
            q(z1|x, z2)   — separate GatedDense paths for x and z2, concatenated
  Decoder:  p(z1|z2)      — GatedDense stack, top-down prior on z1
            p(x|z1, z2)   — separate paths for z1 and z2, concatenated, then to output heads

All prior types (standard, VampPrior, exemplar) work via BaseModel inheritance.
The KL decomposes as:
  KL = KL(q(z1|x,z2) || p(z1|z2)) + KL(q(z2|x) || p(z2))

Most methods (forward, calculate_loss, p_x, q_z1, p_z1, generate_x) live in
AbsHModel — this class only defines the layer architecture via create_model().

See also: convHVAE_2level.py for the convolutional variant.
"""
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from utils.nn import GatedDense, NonLinear
from models.AbsHModel import BaseHModel


class VAE(BaseHModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args):
        """Define the encoder and decoder layers for the 2-level hierarchy.

        Note: output heads (p_x_mean, p_x_logvar) are created by BaseModel.__init__
        before this method is called — they don't need to be defined here.
        """
        print("create_model")

        self.args = args

        # ----- encoder: q(z2 | x) -----
        # Bottom-up: map input to z2 posterior parameters.
        # This is the "top-level" latent — sees only x, not z1.
        self.q_z_layers = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), self.args.hidden_size),
            GatedDense(self.args.hidden_size, self.args.hidden_size)
        )

        self.q_z_mean = Linear(self.args.hidden_size, self.args.z2_size)

        if args.same_variational_var:
            # Scalar variance shared across all z2 dimensions (ablation mode)
            self.q_z_logvar = nn.Parameter(torch.randn((1)))
        else:
            # Per-dimension logvar with HardTanh clamp to prevent
            # posterior collapse (min=-6) and runaway variance (max=2)
            self.q_z_logvar = NonLinear(self.args.hidden_size, self.args.z2_size,
                                        activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ----- encoder: q(z1 | x, z2) -----
        # Bottom-level latent conditioned on both x and z2.
        # Separate paths for x and z2 are concatenated before the joint layers,
        # giving each input its own feature extraction before fusion.
        self.q_z1_layers_x = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), self.args.hidden_size)
        )
        self.q_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, self.args.hidden_size)
        )
        self.q_z1_layers_joint = nn.Sequential(
            GatedDense(2 * self.args.hidden_size, self.args.hidden_size)  # concat of x and z2 paths
        )

        self.q_z1_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.q_z1_logvar = NonLinear(self.args.hidden_size, self.args.z1_size,
                                     activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ----- decoder: p(z1 | z2) -----
        # Top-down prior: z2 generates the prior distribution over z1.
        # The KL for z1 is measured against this learned prior (not N(0,I)).
        self.p_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, self.args.hidden_size),
            GatedDense(self.args.hidden_size, self.args.hidden_size)
        )

        self.p_z1_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.p_z1_logvar = NonLinear(self.args.hidden_size, self.args.z1_size,
                                     activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ----- decoder: p(x | z1, z2) -----
        # Both latent levels contribute to the reconstruction.
        # Separate paths are concatenated, then passed through a joint layer
        # before the output heads (p_x_mean, p_x_logvar from BaseModel).
        self.p_x_layers_z1 = nn.Sequential(
            GatedDense(self.args.z1_size, self.args.hidden_size)
        )
        self.p_x_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, self.args.hidden_size)
        )
        self.p_x_layers_joint = nn.Sequential(
            GatedDense(2 * self.args.hidden_size, self.args.hidden_size)  # concat of z1 and z2 paths
        )
