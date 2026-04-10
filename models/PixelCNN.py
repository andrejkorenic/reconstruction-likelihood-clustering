"""PixelCNN-based hierarchical VAE.

Combines a two-level latent hierarchy (z1, z2) with a PixelSNAIL
autoregressive decoder. The decoder is conditioned on both latent
variables AND the partially-generated image, enabling it to model
fine-grained pixel dependencies that standard decoders miss.

Generation is pixel-by-pixel (raster scan order), making it much
slower than standard VAE generation but producing sharper samples.

Architecture:
    Encoder:  GatedConv2d → z2 → GatedConv2d+z2 → z1
    Decoder:  z1,z2 → reshape to image → PixelSNAIL(x, z1, z2) → output
    Prior:    p(z1|z2) learned MLP, p(z2) standard/vampprior/exemplar

Reference: Gulrajani et al., "PixelVAE" (2016), van den Oord et al.,
"PixelSNAIL" (2017)
"""
from __future__ import print_function
import numpy as np
import torch.nn as nn
from utils.nn import GatedDense, NonLinear, Conv2d, GatedConv2d, PixelSNAIL
from models.AbsHModel import BaseHModel
import torch


class VAE(BaseHModel):
    def __init__(self, args):
        super(VAE, self).__init__(args)

    def create_model(self, args):
        # h_size: flattened feature map dimension after conv encoder
        if args.dataset_name == 'freyfaces':
            self.h_size = 210
        elif args.dataset_name == 'cifar10' or args.dataset_name == 'svhn':
            self.h_size = 384
        else:
            self.h_size = 294   # MNIST-like (28×28)

        # ============================================================
        # Encoder: q(z2 | x) — conv stack → flatten → mean/logvar
        # ============================================================
        self.q_z_layers = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 7, 1, 3),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 5, 1, 2),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        self.q_z_mean = NonLinear(self.h_size, self.args.z2_size, activation=None)
        self.q_z_logvar = NonLinear(self.h_size, self.args.z2_size,
                                   activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ============================================================
        # Encoder: q(z1 | x, z2)
        # ============================================================
        self.q_z1_layers_x = nn.Sequential(
            GatedConv2d(self.args.input_size[0], 32, 3, 1, 1),
            GatedConv2d(32, 32, 3, 2, 1),
            GatedConv2d(32, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 2, 1),
            GatedConv2d(64, 6, 3, 1, 1)
        )
        self.q_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, self.h_size))
        self.q_z1_layers_joint = nn.Sequential(
            GatedDense(2 * self.h_size, 300))
        self.q_z1_mean = NonLinear(300, self.args.z1_size, activation=None)
        self.q_z1_logvar = NonLinear(300, self.args.z1_size,
                                    activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ============================================================
        # Conditional prior: p(z1 | z2)
        # ============================================================
        self.p_z1_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, 300),
            GatedDense(300, 300)
        )
        self.p_z1_mean = NonLinear(300, self.args.z1_size, activation=None)
        self.p_z1_logvar = NonLinear(300, self.args.z1_size,
                                    activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # ============================================================
        # Decoder: p(x | z1, z2) with PixelSNAIL autoregressive head
        # z1,z2 are projected to image-shaped tensors, concatenated
        # with x, and fed through PixelSNAIL for pixel-level modeling
        # ============================================================
        self.p_x_layers_z1 = nn.Sequential(
            GatedDense(self.args.z1_size, np.prod(self.args.input_size)))
        self.p_x_layers_z2 = nn.Sequential(
            GatedDense(self.args.z2_size, np.prod(self.args.input_size)))

        # PixelSNAIL: autoregressive decoder with self-attention
        # Input channels = x + z1 + z2 (3 × input_size[0] for binary MNIST)
        self.pixelcnn = PixelSNAIL([28, 28], 64, 64, 3, 1, 4, 64)

        # --- Output head ---
        if self.args.input_type == 'binary':
            self.p_x_mean = Conv2d(64, 1, 1, 1, 0, activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.p_x_mean = Conv2d(64, self.args.input_size[0], 1, 1, 0,
                                   activation=nn.Sigmoid(), bias=False)
            self.p_x_logvar = Conv2d(64, self.args.input_size[0], 1, 1, 0,
                                     activation=nn.Hardtanh(min_val=-4.5, max_val=0.), bias=False)

    def pixelcnn_generate(self, z1, z2):
        """Autoregressive sampling: generate image pixel-by-pixel.

        Raster-scan order (top-left to bottom-right). At each position,
        the decoder sees all previously generated pixels plus z1, z2.
        Slow (H×W forward passes) but produces sharp samples.

        Args:
            z1: bottom-level latent, shape (batch, z1_size)
            z2: top-level latent, shape (batch, z2_size)

        Returns:
            Generated image means, shape (batch, C, H, W)
        """
        x_zeros = torch.zeros(
            (z1.size(0), self.args.input_size[0],
             self.args.input_size[1], self.args.input_size[2]))
        x_zeros = x_zeros.to(self.args.device)

        for i in range(self.args.input_size[1]):
            for j in range(self.args.input_size[2]):
                samples_mean, samples_logvar = self.p_x(z1, z2, x=x_zeros.detach())
                samples_mean = samples_mean.view(
                    samples_mean.size(0), self.args.input_size[0],
                    self.args.input_size[1], self.args.input_size[2])

                if self.args.input_type == 'binary':
                    # Bernoulli sampling at pixel (i, j)
                    probs = samples_mean[:, :, i, j].data
                    x_zeros[:, :, i, j] = torch.bernoulli(probs).float()
                    samples_gen = samples_mean

                elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
                    # Logistic distribution sampling at pixel (i, j)
                    binsize = 1. / 256.
                    samples_logvar = samples_logvar.view(
                        samples_mean.size(0), self.args.input_size[0],
                        self.args.input_size[1], self.args.input_size[2])
                    means = samples_mean[:, :, i, j].data
                    logvar = samples_logvar[:, :, i, j].data
                    # inverse CDF of logistic: mean + scale * log(u/(1-u))
                    u = torch.rand(means.size()).cuda()
                    y = torch.log(u) - torch.log(1. - u)
                    sample = means + torch.exp(logvar) * y
                    x_zeros[:, :, i, j] = torch.floor(sample / binsize) * binsize
                    samples_gen = samples_mean

        return samples_gen

    def forward(self, x):
        """Reshape flat input to (C, H, W) then delegate to BaseHModel."""
        x = x.view(-1, self.args.input_size[0], self.args.input_size[1], self.args.input_size[2])
        return super(VAE, self).forward(x)


