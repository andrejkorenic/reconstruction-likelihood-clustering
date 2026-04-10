"""
Posterior analysis: true posterior vs variational posterior + SIR.

Visualizes the difference between the true posterior p(z|x) (computed via
Bayes' rule on a 2D grid) and the variational approximation q(z|x) learned
by the encoder. Also performs Sampling Importance Resampling (SIR) to show
how importance-weighted samples better approximate the true posterior.

Requires a pretrained model with z1_size=2.

Usage:
    uv run python posterior_analysis.py --dir pretrained_models/.../

Inspired by the posterior visualization approach in:
    https://github.com/nbip/IWAE (Nicki Skafte Detlefsen)
    Reimplemented from scratch in PyTorch using our own model API.

References:
    * Burda et al. (2015) — Importance Weighted Autoencoders
      https://arxiv.org/abs/1509.00519
    * Cremer et al. (2017) — Reinterpreting Importance-Weighted Autoencoders
      https://arxiv.org/abs/1705.10306
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.load_data.data_loader_instances import load_dataset
from utils.model_io import importing_model, load_model
from utils.distributions import log_normal_diag


# ======================================================================================================================
# Argument parsing
# ======================================================================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Posterior analysis: true vs variational posterior + SIR',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dir', type=str, required=True,
                        help='path to pretrained model directory')
    parser.add_argument('--n_examples', type=int, default=10,
                        help='number of test examples to visualize')
    parser.add_argument('--L', type=int, default=5000,
                        help='number of samples from q(z|x) for SIR')
    parser.add_argument('--n_sir', type=int, default=200,
                        help='number of resampled SIR points')
    parser.add_argument('--grid_size', type=int, default=200,
                        help='resolution of the 2D grid (grid_size x grid_size)')
    parser.add_argument('--scale', type=float, default=3.0,
                        help='how many stds around variational mean for grid range')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    return parser.parse_args()


# ======================================================================================================================
# Core computation
# ======================================================================================================================

@torch.no_grad()
def compute_posteriors(model, x_single, grid_size=200, scale=3.0, L=5000, n_sir=200):
    """
    Compute true posterior, variational posterior, and SIR samples for one input.

    The true posterior is computed via Bayes' rule on a 2D grid:
        log p(z|x) = log p(x|z) + log p(z) - log p(x)
    where log p(x) is approximated by logsumexp normalization over the grid.

    SIR draws L samples from q(z|x), computes importance weights
        w_k = p(x|z_k) * p(z_k) / q(z_k|x)
    and resamples n_sir points proportional to the weights.

    Args:
        model: trained VAE/IWAE with z1_size=2
        x_single: single flattened input, shape (D,)
        grid_size: resolution per axis (total grid points = grid_size^2)
        scale: grid extends scale * std around the variational mean
        L: number of proposal samples for SIR
        n_sir: number of resampled SIR points

    Returns:
        dict with grid coordinates, log-posteriors, samples, and reconstructions
    """
    device = next(model.parameters()).device
    x = x_single.unsqueeze(0).to(device)  # (1, D)

    # ---- 1. Variational posterior parameters from encoder
    q_mu, q_logvar = model.q_z(x)          # each (1, 2)
    q_mu = q_mu.squeeze(0)                  # (2,)
    q_logvar = q_logvar.squeeze(0)          # (2,)
    q_std = torch.exp(0.5 * q_logvar)       # (2,)

    # ---- 2. Build 2D grid around variational mean
    z0_lo = max(-4, (q_mu[0] - scale * q_std[0]).item())
    z0_hi = min(4,  (q_mu[0] + scale * q_std[0]).item())
    z1_lo = max(-4, (q_mu[1] - scale * q_std[1]).item())
    z1_hi = min(4,  (q_mu[1] + scale * q_std[1]).item())

    z0 = torch.linspace(z0_lo, z0_hi, grid_size, device=device)
    z1 = torch.linspace(z1_lo, z1_hi, grid_size, device=device)
    Z0, Z1 = torch.meshgrid(z0, z1, indexing='xy')
    grid = torch.stack([Z0.reshape(-1), Z1.reshape(-1)], dim=1)  # (N, 2)
    N = grid.shape[0]

    # ---- 3. True posterior on grid: log p(z|x) ∝ log p(x|z) + log p(z)
    #
    # Decode each grid point, then evaluate log p(x|z) for the original x.
    x_mean_grid, x_logvar_grid = model.p_x(grid)           # (N, D)
    x_expanded = x.expand(N, -1)                            # (N, D)
    # log p(x|z) — reconstruction log-likelihood
    log_pxz = model.reconstruction_loss(x_expanded, x_mean_grid, x_logvar_grid)  # (N,)

    # log p(z) — prior, supports standard/vampprior via model.log_p_z()
    dummy_indices = torch.zeros(N, 1, dtype=torch.long, device=device)
    log_pz = model.log_p_z(z=(grid, dummy_indices), exemplars_embedding=None)     # (N,)

    # Normalize: log p(z|x) = log p(x|z) + log p(z) - log p(x)
    # where log p(x) ≈ logsumexp over the grid (up to a grid-spacing constant)
    log_posterior_unnorm = log_pxz + log_pz
    log_true_posterior = log_posterior_unnorm - torch.logsumexp(log_posterior_unnorm, dim=0)

    # ---- 4. Variational posterior on grid: log q(z|x)
    log_var_posterior = log_normal_diag(
        grid, q_mu.unsqueeze(0), q_logvar.unsqueeze(0), dim=1
    )  # (N,)

    # ---- 5. SIR (Sampling Importance Resampling)
    #
    # Sample L points from q(z|x), compute importance weights, resample n_sir.
    # We use torch.distributions directly (not model.reparameterize) because
    # we need L independent samples, not the K-sample mechanism from IWAE.
    q_dist = torch.distributions.Normal(q_mu, q_std)
    z_samples = q_dist.rsample((L,))  # (L, 2)

    # log p(x|z) for each SIR sample
    x_mean_sir, x_logvar_sir = model.p_x(z_samples)        # (L, D)
    x_expanded_sir = x.expand(L, -1)                        # (L, D)
    log_pxz_sir = model.reconstruction_loss(
        x_expanded_sir, x_mean_sir, x_logvar_sir
    )  # (L,)

    # log p(z) for each SIR sample
    dummy_indices_sir = torch.zeros(L, 1, dtype=torch.long, device=device)
    log_pz_sir = model.log_p_z(
        z=(z_samples, dummy_indices_sir), exemplars_embedding=None
    )  # (L,)

    # log q(z|x) for each SIR sample
    log_qzx_sir = log_normal_diag(
        z_samples, q_mu.unsqueeze(0), q_logvar.unsqueeze(0), dim=1
    )  # (L,)

    # importance weights: log w_k = log p(x|z) + log p(z) - log q(z|x)
    log_w = log_pxz_sir + log_pz_sir - log_qzx_sir         # (L,)
    w = torch.softmax(log_w, dim=0)                          # (L,)

    # resample according to importance weights
    sir_indices = torch.multinomial(w, n_sir, replacement=False)
    z_sir = z_samples[sir_indices]                           # (n_sir, 2)

    # ---- 6. Reconstructions
    # From variational samples (first n_sir of the L samples)
    x_recon_var = x_mean_sir[:n_sir]                         # (n_sir, D)
    # From SIR samples
    x_recon_sir = x_mean_sir[sir_indices]                    # (n_sir, D)

    return {
        # grid
        'Z0': Z0.cpu().numpy(),
        'Z1': Z1.cpu().numpy(),
        'z0_range': (z0_lo, z0_hi),
        'z1_range': (z1_lo, z1_hi),
        # posteriors on grid
        'log_true_posterior': log_true_posterior.reshape(grid_size, grid_size).cpu().numpy(),
        'log_var_posterior': log_var_posterior.reshape(grid_size, grid_size).cpu().numpy(),
        # samples
        'z_samples': z_samples.cpu().numpy(),       # (L, 2) — all q(z|x) samples
        'z_sir': z_sir.cpu().numpy(),               # (n_sir, 2) — SIR resampled
        # reconstructions
        'x_recon_var': x_recon_var.cpu().numpy(),   # (n_sir, D)
        'x_recon_sir': x_recon_sir.cpu().numpy(),   # (n_sir, D)
    }


# ======================================================================================================================
# Visualization
# ======================================================================================================================

def make_reconstruction_canvas(x_recon, input_size, n=5):
    """Arrange n*n reconstructions into a single image grid."""
    h, w = input_size[1], input_size[2]
    canvas = np.ones((n * h, n * w))
    for j in range(n):
        for k in range(n):
            idx = j * n + k
            if idx < len(x_recon):
                img = x_recon[idx].reshape(input_size)
                canvas[j * h:(j + 1) * h, k * w:(k + 1) * w] = np.clip(img[0], 0, 1)
    return canvas


def plot_posterior_comparison(x_original, results, input_size, save_path):
    """
    Plot 2x4 comparison of true vs variational posterior + SIR.

    Layout:
        Row 0: original | true+var contours | var samples | var reconstructions
        Row 1:  (empty) |          (empty)  | SIR samples | SIR reconstructions
    """
    Z0 = results['Z0']
    Z1 = results['Z1']
    z0_range = results['z0_range']
    z1_range = results['z1_range']
    log_true = results['log_true_posterior']
    log_var = results['log_var_posterior']
    z_samples = results['z_samples']
    z_sir = results['z_sir']

    true_post = np.exp(log_true)
    var_post = np.exp(log_var)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    # ---- (0,0) Original digit
    img = x_original.reshape(input_size)
    if input_size[0] == 1:
        axes[0, 0].imshow(img[0], cmap='gray_r')
    else:
        axes[0, 0].imshow(np.transpose(img, (1, 2, 0)))
    axes[0, 0].set_title('Original', fontsize=16)
    axes[0, 0].axis('off')

    # ---- (0,1) True + variational posterior contours
    axes[0, 1].imshow(
        true_post, cmap='gray_r',
        extent=[z0_range[0], z0_range[1], z1_range[0], z1_range[1]],
        origin='lower', aspect='auto',
    )
    axes[0, 1].contour(Z0, Z1, true_post + 1e-16, 5, cmap='RdGy_r', linewidths=2)
    axes[0, 1].contour(Z0, Z1, var_post + 1e-16, 5, cmap='Purples', linewidths=2)
    axes[0, 1].set_xlim(z0_range)
    axes[0, 1].set_ylim(z1_range)
    axes[0, 1].set_title('True (red) + Var (purple)\nposterior', fontsize=14)

    # ---- (0,2) Variational posterior samples on true posterior background
    n_scatter = min(500, len(z_samples))
    axes[0, 2].imshow(
        true_post, cmap='gray_r',
        extent=[z0_range[0], z0_range[1], z1_range[0], z1_range[1]],
        origin='lower', aspect='auto',
    )
    axes[0, 2].contour(Z0, Z1, var_post + 1e-16, 5, cmap='Purples', linewidths=2)
    axes[0, 2].scatter(
        z_samples[:n_scatter, 0], z_samples[:n_scatter, 1],
        marker='.', alpha=0.3, color='purple', s=10,
    )
    axes[0, 2].set_xlim(z0_range)
    axes[0, 2].set_ylim(z1_range)
    axes[0, 2].set_title('Variational samples', fontsize=14)

    # ---- (0,3) Reconstructions from variational samples
    canvas_var = make_reconstruction_canvas(results['x_recon_var'], input_size)
    axes[0, 3].imshow(canvas_var, cmap='gray_r')
    axes[0, 3].axis('off')
    axes[0, 3].set_title('Var reconstructions', fontsize=14)

    # ---- (1,0) empty
    axes[1, 0].axis('off')

    # ---- (1,1) empty
    axes[1, 1].axis('off')

    # ---- (1,2) SIR samples on true posterior background
    axes[1, 2].imshow(
        true_post, cmap='gray_r',
        extent=[z0_range[0], z0_range[1], z1_range[0], z1_range[1]],
        origin='lower', aspect='auto',
    )
    axes[1, 2].contour(Z0, Z1, var_post + 1e-16, 5, cmap='Purples', linewidths=2)
    axes[1, 2].scatter(
        z_sir[:, 0], z_sir[:, 1],
        marker='.', alpha=0.3, color='purple', s=10,
    )
    axes[1, 2].set_xlim(z0_range)
    axes[1, 2].set_ylim(z1_range)
    axes[1, 2].set_title('SIR samples', fontsize=14)

    # ---- (1,3) Reconstructions from SIR samples
    canvas_sir = make_reconstruction_canvas(results['x_recon_sir'], input_size)
    axes[1, 3].imshow(canvas_sir, cmap='gray_r')
    axes[1, 3].axis('off')
    axes[1, 3].set_title('SIR reconstructions', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ======================================================================================================================
# Main
# ======================================================================================================================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Load model (analysis.py pattern: config + checkpoint)
    snap_dir = args.dir
    if not snap_dir.endswith('/'):
        snap_dir += '/'

    # parse model_name from parent folder: "..._model_name=vae" → "vae"
    parent_folder = os.path.basename(os.path.dirname(snap_dir.rstrip('/')))
    marker = 'model_name='
    marker_idx = parent_folder.find(marker)
    if marker_idx == -1:
        raise ValueError(
            f"Cannot parse model_name from folder '{parent_folder}'. "
            f"Expected 'model_name=<name>' in the parent folder name."
        )
    model_name = parent_folder[marker_idx + len(marker):]

    config = torch.load(
        os.path.join(snap_dir, f'{model_name}.config'),
        map_location=device, weights_only=False,
    )
    config.device = device

    # ---- Guard: z1_size must be 2 for grid visualization
    if config.z1_size != 2:
        raise ValueError(
            f"Posterior analysis requires z1_size=2 (got {config.z1_size}). "
            f"Train a model with --z1_size 2 for this visualization."
        )

    if config.prior == 'exemplar_prior':
        raise NotImplementedError(
            "Posterior analysis does not yet support exemplar_prior. "
            "Use standard or vampprior."
        )

    VAE = importing_model(config)
    model = VAE(config)
    model.to(device)
    load_model(os.path.join(snap_dir, 'checkpoint_best.pth'), model)
    model.eval()

    print(f'Loaded {model_name} model from {snap_dir}')
    print(f'  prior={config.prior}, z1_size={config.z1_size}, input_type={config.input_type}')

    # ---- Load test data
    train_loader, val_loader, test_loader, config = load_dataset(
        config, no_binarization=True,
    )

    # Collect test images
    test_data = []
    for batch in test_loader:
        if len(batch) == 3:
            data, _, _ = batch
        else:
            data, _ = batch
        test_data.append(data)
    test_data = torch.cat(test_data, dim=0)

    n_examples = min(args.n_examples, len(test_data))
    print(f'Analyzing {n_examples} test examples...')

    # ---- Output directory
    out_dir = os.path.join(snap_dir, 'posterior_analysis')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Run analysis
    for i in range(n_examples):
        x = test_data[i]
        results = compute_posteriors(
            model, x,
            grid_size=args.grid_size,
            scale=args.scale,
            L=args.L,
            n_sir=args.n_sir,
        )
        save_path = os.path.join(out_dir, f'posterior_{i}.png')
        plot_posterior_comparison(x.numpy(), results, config.input_size, save_path)
        print(f'  [{i+1}/{n_examples}] saved {save_path}')

    print(f'Done. Results saved to {out_dir}')


if __name__ == '__main__':
    main()
