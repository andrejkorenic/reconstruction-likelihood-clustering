"""Convolutional two-level hierarchical VAE.

Architecture: GatedConv2d encoder → (z2, z1) → GatedConvTranspose2d decoder.
Same graphical model as HVAE_2level but with convolutional layers
throughout, making it suitable for spatially-structured data.

Encoder: stride-2 GatedConv2d downsamples to (6, H//4, W//4), then flattened.
Decoder: GatedDense projects to (64, H//4, W//4) spatial bottleneck, then
GatedConvTranspose2d upsamples back to full resolution (symmetric with ConvVAE).
"""
from __future__ import print_function
import numpy as np
import torch.nn as nn
from utils.nn import GatedDense, NonLinear, Conv2d, GatedConv2d, GatedConvTranspose2d
from models.AbsHModel import BaseHModel


class VAE(BaseHModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args):
        C, H, W = args.input_size
        # h_size: flattened feature map after conv encoder (6 output channels, 2x stride-2 downsamples)
        self.h_size = 6 * (H // 4) * (W // 4)

        fc_size = 300  # dense bottleneck width

        # ============================================================
        # Encoder: q(z2 | x)
        # Conv stack → flatten → mean/logvar
        # ============================================================
        self.q_z_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 7, 1, 3, no_attention=args.no_attention),
            GatedConv2d(32, 32, 3, 2, 1, no_attention=args.no_attention),   # downsample ×2
            GatedConv2d(32, 64, 5, 1, 2, no_attention=args.no_attention),
            GatedConv2d(64, 64, 3, 2, 1, no_attention=args.no_attention),   # downsample ×2
            GatedConv2d(64, 6, 3, 1, 1, no_attention=args.no_attention)
        )
        self.q_z_mean = NonLinear(self.h_size, self.args.z2_size, activation=None)
        self.q_z_logvar = NonLinear(self.h_size, self.args.z2_size,
                                   activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ============================================================
        # Encoder: q(z1 | x, z2) — processes x and z2 separately,
        # then jointly to produce z1 parameters
        # ============================================================
        self.q_z1_layers_x = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 3, 1, 1, no_attention=args.no_attention),
            GatedConv2d(32, 32, 3, 2, 1, no_attention=args.no_attention),
            GatedConv2d(32, 64, 3, 1, 1, no_attention=args.no_attention),
            GatedConv2d(64, 64, 3, 2, 1, no_attention=args.no_attention),
            GatedConv2d(64, 6, 3, 1, 1, no_attention=args.no_attention)
        )
        self.q_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, self.h_size))
        self.q_z1_layers_joint = nn.Sequential(
            GatedDense(2 * self.h_size, fc_size))
        self.q_z1_mean = NonLinear(fc_size, self.args.z1_size, activation=None)
        self.q_z1_logvar = NonLinear(fc_size, self.args.z1_size,
                                    activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ============================================================
        # Conditional prior: p(z1 | z2)
        # ============================================================
        self.p_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, fc_size, no_attention=args.no_attention),
            GatedDense(fc_size, fc_size, no_attention=args.no_attention)
        )
        self.p_z1_mean = NonLinear(fc_size, self.args.z1_size, activation=None)
        self.p_z1_logvar = NonLinear(fc_size, self.args.z1_size,
                                    activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ============================================================
        # Decoder: p(x | z1, z2)
        # z1,z2 → dense → reshape to image → conv stack → output head
        # ============================================================
        self.p_x_layers_z1 = nn.Sequential(
            GatedDense(self.args.z1_size, fc_size, no_attention=args.no_attention))
        self.p_x_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, fc_size, no_attention=args.no_attention))

        # Decoder spatial bottleneck: project to (64, H//4, W//4) then upsample
        # (replaces dense bridge that projected directly to full C*H*W)
        self.decoder_h_size = 64 * (H // 4) * (W // 4)
        self.p_x_layers_joint_pre = nn.Sequential(
            GatedDense(2 * fc_size, self.decoder_h_size, no_attention=args.no_attention))

        # Transposed conv upsampling (symmetric with encoder stride-2 downsamples)
        self.p_x_layers_joint = nn.Sequential(
            GatedConvTranspose2d(64, 64, 4, 2, 1, no_attention=args.no_attention),  # upsample x2
            GatedConvTranspose2d(64, 32, 4, 2, 1, no_attention=args.no_attention),  # upsample x2
            GatedConv2d(32, 32, 5, 1, 2, no_attention=args.no_attention),           # refinement
        )

        # --- Output head (depends on input_type) ---
        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(32, C, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type in ('gray', 'continuous'):
            self.p_x_mean = Conv2d(32, C, 1, 1, 0)
            self.p_x_logvar = Conv2d(32, C, 1, 1, 0,
                                     activation=nn.Hardtanh(min_val=-4.5, max_val=0.))

    def forward(self, x):
        """Reshape flat input to (C, H, W) then delegate to BaseHModel."""
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        return super(VAE, self).forward(x)

