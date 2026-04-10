"""
TimeVAE — Variational Autoencoder for multivariate time series.

Encoder: Conv1d with stride-2 downsampling.
Decoder: interpretable additive decomposition:
  reconstruction = Level + Trend + Seasonal + Residual

All prior types (standard, VampPrior, exemplar) work out of the box
via BaseModel inheritance. VampPrior pseudo-inputs become learnable
prototypical time series patterns.

References:
  * Desai, A., Freeman, C., Wang, Z., & Beaver, I. (2021).
    TimeVAE: A Variational Auto-Encoder for Multivariate Time Series
    Generation. https://arxiv.org/abs/2111.08095
"""

from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from utils.nn import NonLinear
from utils.distributions import log_normal_diag, log_beta
from models.AbsModel import AbsModel


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class _TimeSeriesEncoder(nn.Module):
    """Conv1d encoder that accepts flat input and reshapes internally."""

    def __init__(self, feat_dim, seq_len, hidden_sizes, out_dim):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len

        layers = []
        in_ch = feat_dim
        for h in hidden_sizes:
            layers.append(nn.Conv1d(in_ch, h, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = h
        self.convs = nn.Sequential(*layers)

        # compute flattened dim with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, feat_dim, seq_len)
            conv_out = self.convs(dummy)
            self._flat_dim = conv_out.numel()

        self.fc = nn.Linear(self._flat_dim, out_dim)

    def forward(self, x):
        x = x.view(-1, self.feat_dim, self.seq_len)
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return torch.relu(self.fc(x))


# --------------------------------------------------------------------------- #
# Decoder components
# --------------------------------------------------------------------------- #
class _LevelModel(nn.Module):
    """Constant baseline broadcast across all timesteps."""

    def __init__(self, latent_dim, feat_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, feat_dim),
        )

    def forward(self, z):
        level = self.net(z)                         # (batch, feat_dim)
        return level.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, feat_dim)


class _TrendLayer(nn.Module):
    """Polynomial trend over normalized time axis [0, 1]."""

    def __init__(self, latent_dim, feat_dim, seq_len, trend_poly):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.trend_poly = trend_poly

        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, feat_dim * trend_poly),
        )

        # polynomial basis: t^1, t^2, ..., t^trend_poly
        t = torch.linspace(0, 1, seq_len)
        basis = torch.stack([t ** (p + 1) for p in range(trend_poly)], dim=0)  # (trend_poly, seq_len)
        self.register_buffer('basis', basis)

    def forward(self, z):
        params = self.net(z)                              # (batch, feat_dim * trend_poly)
        params = params.view(-1, self.feat_dim, self.trend_poly)  # (batch, feat_dim, trend_poly)
        trend = torch.einsum('bfp,ps->bsf', params, self.basis)  # (batch, seq_len, feat_dim)
        return trend


class _SeasonalLayer(nn.Module):
    """Learned seasonal patterns with configurable periods."""

    def __init__(self, latent_dim, feat_dim, seq_len, custom_seas):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas  # list of (num_seasons, season_len)

        self.nets = nn.ModuleList()
        self.season_indices = []
        for num_seasons, season_len in custom_seas:
            self.nets.append(nn.Sequential(
                nn.Linear(latent_dim, latent_dim), nn.ReLU(),
                nn.Linear(latent_dim, feat_dim * num_seasons),
            ))
            # precompute which season each timestep belongs to
            idx = torch.arange(seq_len) // season_len % num_seasons
            self.season_indices.append(idx)

    def forward(self, z):
        batch = z.size(0)
        total = torch.zeros(batch, self.seq_len, self.feat_dim, device=z.device)

        for net, idx in zip(self.nets, self.season_indices):
            params = net(z).view(batch, self.feat_dim, -1)   # (batch, feat_dim, num_seasons)
            idx_dev = idx.to(z.device)
            # gather: for each timestep pick the right season parameter
            idx_exp = idx_dev.unsqueeze(0).unsqueeze(0).expand(batch, self.feat_dim, -1)  # (batch, feat_dim, seq_len)
            seasonal = torch.gather(params, 2, idx_exp)       # (batch, feat_dim, seq_len)
            total += seasonal.permute(0, 2, 1)                # (batch, seq_len, feat_dim)

        return total


class _ResidualConnection(nn.Module):
    """ConvTranspose1d upsampling to capture residual temporal patterns."""

    def __init__(self, latent_dim, feat_dim, seq_len, hidden_sizes):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len

        # mirror the encoder: upsample with ConvTranspose1d
        reversed_sizes = list(reversed(hidden_sizes))

        # project latent to the encoder's last conv output shape
        # compute what the encoder's last conv output length would be
        L = seq_len
        for _ in hidden_sizes:
            L = (L + 2 * 1 - 3) // 2 + 1  # Conv1d(k=3, s=2, p=1)
        self._start_len = L
        self._start_ch = reversed_sizes[0]
        self.fc = nn.Linear(latent_dim, self._start_ch * L)

        layers = []
        in_ch = reversed_sizes[0]
        for h in reversed_sizes[1:]:
            layers.append(nn.ConvTranspose1d(in_ch, h, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU())
            in_ch = h
        # final layer maps back to feat_dim
        layers.append(nn.ConvTranspose1d(in_ch, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.convs = nn.Sequential(*layers)

    def forward(self, z):
        h = torch.relu(self.fc(z))
        h = h.view(-1, self._start_ch, self._start_len)
        h = self.convs(h)                                 # (batch, feat_dim, ~seq_len)
        # crop or pad to exact seq_len (ConvTranspose1d may overshoot)
        if h.size(2) > self.seq_len:
            h = h[:, :, :self.seq_len]
        elif h.size(2) < self.seq_len:
            h = nn.functional.pad(h, (0, self.seq_len - h.size(2)))
        return h.permute(0, 2, 1)                         # (batch, seq_len, feat_dim)


# --------------------------------------------------------------------------- #
# TimeVAE decoder: additive decomposition
# --------------------------------------------------------------------------- #
class _TimeVAEDecoder(nn.Module):
    """Level + Trend + Seasonal + Residual additive decoder."""

    def __init__(self, latent_dim, feat_dim, seq_len, hidden_sizes,
                 trend_poly=0, custom_seas=None, use_residual=True):
        super().__init__()
        self.level = _LevelModel(latent_dim, feat_dim, seq_len)

        self.trend = None
        if trend_poly > 0:
            self.trend = _TrendLayer(latent_dim, feat_dim, seq_len, trend_poly)

        self.seasonal = None
        if custom_seas:
            self.seasonal = _SeasonalLayer(latent_dim, feat_dim, seq_len, custom_seas)

        self.residual = None
        if use_residual:
            self.residual = _ResidualConnection(latent_dim, feat_dim, seq_len, hidden_sizes)

    def forward(self, z):
        out = self.level(z)
        if self.trend is not None:
            out = out + self.trend(z)
        if self.seasonal is not None:
            out = out + self.seasonal(z)
        if self.residual is not None:
            out = out + self.residual(z)
        return out                                         # (batch, seq_len, feat_dim)

    def decompose(self, z):
        """Return individual decoder components as a dict.

        Each value is (batch, seq_len, feat_dim) or None if disabled.
        """
        parts = {}
        parts['level'] = self.level(z)
        parts['trend'] = self.trend(z) if self.trend is not None else None
        parts['seasonal'] = self.seasonal(z) if self.seasonal is not None else None
        parts['residual'] = self.residual(z) if self.residual is not None else None
        parts['total'] = self.forward(z)
        return parts


# --------------------------------------------------------------------------- #
# Per-timestep variance head
# --------------------------------------------------------------------------- #
class _VarianceHead(nn.Module):
    """Per-timestep log-variance (or log-concentration) prediction from latent z.

    Simple 2-layer MLP: z → Linear(z_dim, 64) → ReLU → Linear(64, seq_len * feat_dim).
    Clamping is applied externally in p_x() since the range depends on the
    reconstruction distribution (Beta vs Gaussian).
    """

    def __init__(self, z_dim, seq_len, feat_dim):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, seq_len * feat_dim),
        )

    def forward(self, z):
        h = self.net(z)                                    # (batch, seq_len * feat_dim)
        return h.reshape(-1, self.seq_len, self.feat_dim)  # (batch, seq_len, feat_dim)


# --------------------------------------------------------------------------- #
# TimeSeriesVAE model
# --------------------------------------------------------------------------- #
class TimeSeriesVAE(AbsModel):
    def __init__(self, args):
        super(TimeSeriesVAE, self).__init__(args)

    def create_model(self, args):
        seq_len = args.seq_len
        feat_dim = args.feat_dim
        h = args.hidden_size
        hidden_sizes = [h // 4, h // 2, h]  # e.g. [50, 100, 200] for h=200

        # encoder: Conv1d
        self.q_z_layers = _TimeSeriesEncoder(feat_dim, seq_len, hidden_sizes, h)
        self.q_z_mean = Linear(h, args.z1_size)
        self.q_z_logvar = NonLinear(h, args.z1_size,
                                    activation=nn.Hardtanh(min_val=-6., max_val=2.))

        # decoder: additive decomposition
        self.decoder = _TimeVAEDecoder(
            latent_dim=args.z1_size,
            feat_dim=feat_dim,
            seq_len=seq_len,
            hidden_sizes=hidden_sizes,
            trend_poly=getattr(args, 'trend_poly', 0),
            custom_seas=getattr(args, 'custom_seas', None),
            use_residual=getattr(args, 'use_residual', True),
        )

        # reconstruction distribution: beta (default for TS) or gaussian
        self.reconstruction_dist = getattr(args, 'reconstruction_dist', 'beta')

        # per-timestep variance head: predicts second distribution parameter at each timestep
        self.variance_head = _VarianceHead(args.z1_size, seq_len, feat_dim)
        if self.reconstruction_dist == 'beta':
            self._concentration_clamp = nn.Hardtanh(min_val=0.5, max_val=8.0)

    def p_x(self, z):
        """Decode latent z to time series via additive decomposition.

        Returns:
            x_mean:  (batch, D) — sigmoid-bounded mean, D = seq_len * feat_dim.
            x_param2: (batch, D) — per-timestep second parameter:
                      Beta mode: clamped log-concentration in [0.5, 8.0].
                      Gaussian mode: clamped log-variance in [-4.5, 0.0].
        """
        z = z.reshape(-1, z.shape[-1])   # flatten (B, N, z_dim) → (B*N, z_dim)
        x_recon = self.decoder(z)                          # (batch, seq_len, feat_dim)
        x_flat = x_recon.reshape(-1, np.prod(self.args.input_size))
        x_mean = torch.sigmoid(x_flat)

        # per-timestep variance
        var_raw = self.variance_head(z)                    # (batch, seq_len, feat_dim)
        var_flat = var_raw.reshape(-1, np.prod(self.args.input_size))

        if self.reconstruction_dist == 'beta':
            x_param2 = self._concentration_clamp(var_flat)  # [0.5, 8.0] per timestep
        else:
            x_param2 = var_flat.clamp(-4.5, 0.0)            # logvar per timestep

        return x_mean, x_param2

    def reconstruction_loss(self, x, x_mean, x_logvar):
        """Reconstruction log-likelihood for continuous time series."""
        if self.reconstruction_dist == 'beta':
            return log_beta(x, x_mean, x_logvar, dim=1)
        else:
            return log_normal_diag(x, x_mean, x_logvar, dim=1)

    def generate_x_from_z(self, z, with_reparameterize=True):
        """Generate time series — no logit_inverse needed."""
        generated_x, _ = self.p_x(z)
        return generated_x

    def decompose_reconstruction(self, x):
        """Encode x, decode z_mean, return per-component decomposition."""
        z_mean, _ = self.q_z(x)
        return self.decoder.decompose(z_mean)
