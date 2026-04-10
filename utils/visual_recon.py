"""Reconstruction likelihood visualization.

For each class in the test set, finds the samples with highest and lowest
log p(x|z) and saves side-by-side grids of originals vs reconstructions.
"""
import logging

import numpy as np
import torch
from pathlib import Path

from utils.plot_images import plot_images

log = logging.getLogger(__name__)

_N_SAMPLES = 5


def plot_reconstruction_likelihood(args, model, test_loader, output_dir):
    """Plot best/worst reconstructions per class by log p(x|z).

    Args:
        args:        Namespace with input_size, input_type, K, device.
        model:       Trained VAE model (subclass of BaseModel).
        test_loader: DataLoader for test split.
        output_dir:  Path where recon_likelihood/ subfolder will be created.
    """
    out_dir = Path(output_dir) / 'recon_likelihood'
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Reconstruction likelihood visualization...")

    original_K = getattr(args, 'K', 1)
    model_K = getattr(model.args, 'K', 1)
    args.K = 1
    model.args.K = 1

    try:
        data, labels, recons, log_likelihoods = _encode_decode_all(
            args, model, test_loader)
    finally:
        args.K = original_K
        model.args.K = model_K

    # TimeSeriesVAE: line plot visualizations (best/worst per class or overall)
    if len(args.input_size) == 2:
        from utils.plot_timeseries import plot_ts_reconstruction
        seq_len = args.input_size[1]
        # flatten (N, feat_dim, seq_len) → (N, feat_dim*seq_len) for plot_ts_reconstruction
        data_flat = data.reshape(len(data), -1)
        recons_flat = recons.reshape(len(recons), -1)

        unique_labels = np.unique(labels)
        has_classes = len(unique_labels) > 1

        if has_classes:
            for label in unique_labels:
                mask = labels == label
                _plot_ts_best_worst(
                    data_flat[mask], recons_flat[mask], log_likelihoods[mask],
                    str(out_dir) + '/', f'class_{label}', seq_len,
                    getattr(args, 'plot_timesteps', None))
                log.info("  TS class %s: best/worst %d pairs",
                         label, min(_N_SAMPLES, int(mask.sum())))
        else:
            _plot_ts_best_worst(
                data_flat, recons_flat, log_likelihoods,
                str(out_dir) + '/', 'overall', seq_len,
                getattr(args, 'plot_timesteps', None))
            log.info("  TS overall: best/worst %d pairs",
                     min(_N_SAMPLES, len(data_flat)))
        log.info("  Saved to %s", out_dir)
        return

    unique_labels = np.unique(labels)
    has_classes = len(unique_labels) > 1

    if has_classes:
        for label in unique_labels:
            mask = labels == label
            _plot_best_worst(
                args, data[mask], recons[mask], log_likelihoods[mask],
                str(out_dir) + '/', f'class_{label}')
            log.info("  Plotting class %s: best/worst %d pairs",
                     label, min(_N_SAMPLES, int(mask.sum())))
    else:
        _plot_best_worst(
            args, data, recons, log_likelihoods,
            str(out_dir) + '/', 'overall')
        log.info("  Plotting overall: best/worst %d pairs",
                 min(_N_SAMPLES, len(data)))

    log.info("  Saved to %s", out_dir)


def _encode_decode_all(args, model, test_loader):
    """Run model.forward() on all test data, compute per-sample log p(x|z).

    Uses model.forward() which correctly handles all model types:
    flat VAE, ConvVAE, hierarchical (HVAE, convHVAE, PixelCNN), IWAE,
    and TimeSeriesVAE.

    Returns:
        data:             numpy array (N, *input_size)
        labels:           numpy array (N,)
        recons:           numpy array (N, *input_size)
        log_likelihoods:  numpy array (N,)
    """
    all_data, all_labels = [], []
    all_recons, all_ll = [], []

    with torch.no_grad():
        for batch in test_loader:
            x, lbl = batch[0], batch[1]
            x = x.to(args.device)

            # model.forward() handles all model types correctly:
            # flat VAE, ConvVAE (auto-reshape), hierarchical (z1+z2),
            # IWAE (K=1 set by caller), TimeSeriesVAE (additive decoder)
            x_mean, x_logvar, _ = model.forward(x)

            ll = model.reconstruction_loss(x, x_mean, x_logvar)

            all_data.append(x.cpu().numpy().reshape(-1, *args.input_size))
            all_labels.append(lbl.numpy())
            all_recons.append(x_mean.cpu().numpy().reshape(-1, *args.input_size))
            all_ll.append(ll.cpu().numpy())

    return (
        np.concatenate(all_data),
        np.concatenate(all_labels).astype(int),
        np.concatenate(all_recons),
        np.concatenate(all_ll),
    )


def _plot_best_worst(args, data, recons, log_likelihoods, out_dir, name_prefix):
    """Sort by likelihood and save best/worst grids.

    Each grid: _N_SAMPLES rows x 2 columns (original | reconstruction).
    """
    n = min(_N_SAMPLES, len(data))
    sorted_idx = np.argsort(log_likelihoods)

    worst_idx = sorted_idx[:n]
    best_idx = sorted_idx[-n:][::-1]

    _save_paired_grid(args, data[best_idx], recons[best_idx],
                      out_dir, f'{name_prefix}_best', n)
    _save_paired_grid(args, data[worst_idx], recons[worst_idx],
                      out_dir, f'{name_prefix}_worst', n)


def _save_paired_grid(args, originals, reconstructions, out_dir, filename, n):
    """Interleave originals and reconstructions into a 2-column grid and save.

    Layout: row i -> [original_i, reconstruction_i]
    plot_images() uses GridSpec(size_x, size_y) so size_x=rows, size_y=cols.
    """
    # Interleave: [orig_0, recon_0, orig_1, recon_1, ...]
    paired = np.stack([originals[:n], reconstructions[:n]], axis=1)
    paired = paired.reshape(2 * n, *args.input_size)

    # size_x = rows (n pairs), size_y = columns (2: original + reconstruction)
    plot_images(args, paired, out_dir, filename, size_x=n, size_y=2)


def _plot_ts_best_worst(data_flat, recons_flat, log_likelihoods, out_dir,
                        name_prefix, seq_len, plot_timesteps):
    """Sort time series by likelihood and save best/worst line-plot grids."""
    from utils.plot_timeseries import plot_ts_reconstruction
    n = min(_N_SAMPLES, len(data_flat))
    sorted_idx = np.argsort(log_likelihoods)
    worst_idx = sorted_idx[:n]
    best_idx = sorted_idx[-n:][::-1]
    plot_ts_reconstruction(
        data_flat[best_idx], recons_flat[best_idx],
        out_dir, f'{name_prefix}_best', seq_len=seq_len,
        n_samples=n, plot_timesteps=plot_timesteps)
    plot_ts_reconstruction(
        data_flat[worst_idx], recons_flat[worst_idx],
        out_dir, f'{name_prefix}_worst', seq_len=seq_len,
        n_samples=n, plot_timesteps=plot_timesteps)
