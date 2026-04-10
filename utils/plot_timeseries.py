import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_ts_reconstruction(original, reconstructed, dir, file_name,
                           seq_len, n_samples=6, plot_timesteps=None):
    """Overlay plot: original (blue) + reconstruction (orange).

    Args:
        original: numpy array (n_samples, feat_dim * seq_len), flat format.
        reconstructed: numpy array (n_samples, feat_dim * seq_len), flat format.
        dir: output directory path (with trailing slash).
        file_name: filename without extension.
        seq_len: sequence length (used to slice flat vector for univariate).
        n_samples: number of subplot rows.
        plot_timesteps: max timesteps to show (None = all).
    """
    t = min(seq_len, plot_timesteps) if plot_timesteps else seq_len
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2 * n_samples),
                             sharex=True)
    if n_samples == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(original[i, :t], color='tab:blue', label='Original')
        ax.plot(reconstructed[i, :t], color='tab:orange',
                label='Reconstruction')
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
        ax.set_ylabel(f'#{i+1}', fontsize=8)
    axes[-1].set_xlabel('Timestep')
    fig.tight_layout()
    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)


def plot_ts_generation(generated, dir, file_name, seq_len, n_samples=6,
                       plot_timesteps=None):
    """Plot generated time series (no reference comparison).

    Also used for plotting originals-only (real.png reference).

    Args:
        generated: numpy array (n_samples, feat_dim * seq_len), flat format.
        dir: output directory path (with trailing slash).
        file_name: filename without extension.
        seq_len: sequence length.
        n_samples: number of subplots.
        plot_timesteps: max timesteps to show (None = all).
    """
    t = min(seq_len, plot_timesteps) if plot_timesteps else seq_len
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 2 * n_samples),
                             sharex=True)
    if n_samples == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(generated[i, :t], color='tab:green')
        ax.set_ylabel(f'#{i+1}', fontsize=8)
    axes[-1].set_xlabel('Timestep')
    fig.tight_layout()
    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)


def plot_ts_decomposition(components, dir, file_name, seq_len, sample_idx=0,
                          plot_timesteps=None):
    """Plot decoder decomposition: Level, Trend, Seasonal, Residual, Total.

    Args:
        components: dict from decoder.decompose().
            Values are numpy arrays of shape (batch, seq_len, feat_dim)
            or None for disabled components.
        dir: output directory path (with trailing slash).
        file_name: filename without extension.
        seq_len: sequence length.
        sample_idx: which sample from the batch to plot.
        plot_timesteps: max timesteps to show (None = all).
    """
    t = min(seq_len, plot_timesteps) if plot_timesteps else seq_len
    order = ['level', 'trend', 'seasonal', 'residual', 'total']
    active = [(name, components[name]) for name in order
              if components.get(name) is not None]

    n = len(active)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, arr) in zip(axes, active):
        # arr is (batch, seq_len, feat_dim) — take sample_idx, squeeze feat_dim
        series = arr[sample_idx, :t, 0]
        lw = 2.0 if name == 'total' else 1.0
        color = 'tab:red' if name == 'total' else 'tab:blue'
        ax.plot(series, color=color, linewidth=lw)
        ax.set_ylabel(name.capitalize(), fontsize=9)

    axes[-1].set_xlabel('Timestep')
    fig.suptitle(f'Decoder Decomposition (sample {sample_idx})', fontsize=11)
    fig.tight_layout()
    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)
