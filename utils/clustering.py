"""
Latent space clustering analysis for VAE models.

Evaluates how well the learned latent space separates true labels using:
  1. Raw latent space clustering (LOO-kNN, K-Means, HDBSCAN)
  2. t-SNE embedding + clustering
  3. UMAP embedding + clustering
  4. Reconstruction probability scatter plots (best/worst per class)

Typical usage (called from final_evaluation with --cluster flag):

    >>> from utils.clustering import cluster_latent
    >>> cluster_latent(args, model, test_data, test_labels, output_dir="results/")

Outputs: SVG scatter plots, PNG reconstruction grids, metrics to logger.
"""
import logging
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    fowlkes_mallows_score,
    v_measure_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors
import sklearn.cluster as cluster
import hdbscan
import umap.umap_ as umap

from utils.distributions import log_bernoulli, log_logistic_256
from utils.plot_images import plot_images

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _scatter_plot(
    X: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    cmap: str = "tab10",
) -> None:
    """
    Generate and save a publication-quality 2-D scatter plot.

    Supports both categorical labels (discrete classes → legend) and
    continuous values (e.g., reconstruction probability → colorbar).
    """
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (N, 2).")
    if len(X) != len(labels):
        raise ValueError("X and labels must contain the same number of samples.")

    df = pd.DataFrame(X, columns=["x", "y"])

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(4, 4))

    # detect continuous vs categorical labels
    is_continuous = (
        np.issubdtype(labels.dtype, np.floating)
        and len(np.unique(labels)) > 20
    )

    if is_continuous:
        df["value"] = labels
        sns.scatterplot(
            data=df, x="x", y="y", hue="value",
            palette=cmap, s=6, alpha=0.7, linewidth=0,
            ax=ax, legend=False,
        )
        norm = plt.Normalize(labels.min(), labels.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.8)
    else:
        df["label"] = pd.Categorical(
            labels.astype(int),
            categories=sorted(np.unique(labels.astype(int))),
            ordered=False,
        )
        sns.scatterplot(
            data=df, x="x", y="y", hue="label",
            palette=cmap, s=6, alpha=0.7, linewidth=0, ax=ax,
            legend=False,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    sns.despine(ax=ax, left=True, bottom=True)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path.with_suffix('.png'), dpi=600, bbox_inches="tight")
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches="tight", format='svg')
    plt.close(fig)


def _loo_knn(
    embedding_data: np.ndarray,
    true_labels: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
) -> np.ndarray:
    """Leave-One-Out k-Nearest Neighbour classification."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(embedding_data)
    _, indices = nn.kneighbors(embedding_data)
    neighbour_labels = true_labels[indices[:, 1:]]
    preds, _ = mode(neighbour_labels, axis=1)
    return preds.ravel()


def _heuristic_labeling(true_labels, cluster_labels):
    """Assign the most frequent true label to each cluster."""
    true_labels = np.asarray(true_labels)
    cluster_labels = np.asarray(cluster_labels)
    new_labels = np.empty_like(true_labels)
    for cl in np.unique(cluster_labels):
        if cl >= 0:  # skip HDBSCAN noise label (-1)
            mask = cluster_labels == cl
            most_common = Counter(true_labels[mask]).most_common(1)[0][0]
            new_labels[mask] = most_common
    return new_labels


def _evaluate_and_log_unsupervised(embedding_data, true_labels, new_labels, name,
                                    has_labels=True):
    """Compute and log clustering metrics. Returns dict of metric values."""
    metrics = {}
    msg = (
        "----------------------------------------\n"
        f"{name} clustering metrics:\n"
    )
    if has_labels:
        metrics["ARI"] = adjusted_rand_score(true_labels, new_labels)
        metrics["AMI"] = adjusted_mutual_info_score(true_labels, new_labels)
        msg += (
            f"Adjusted Rand Score           = {metrics['ARI']:.4f}\n"
            f"Adjusted Mutual Info Score    = {metrics['AMI']:.4f}\n"
        )
    metrics["SS"] = silhouette_score(embedding_data, new_labels)
    metrics["DBI"] = davies_bouldin_score(embedding_data, new_labels)
    metrics["CHI"] = calinski_harabasz_score(embedding_data, new_labels)
    msg += (
        f"Silhouette Score              = {metrics['SS']:.4f}\n"
        f"Davies-Bouldin Index          = {metrics['DBI']:.4f}\n"
        f"Calinski-Harabasz Index       = {metrics['CHI']:.4f}"
    )
    log.info(msg)
    return metrics


def _evaluate_and_log_supervised(embedding_data, true_labels, new_labels):
    """Compute and log supervised classification metrics. Returns dict of metric values."""
    metrics = {
        "ACC": accuracy_score(true_labels, new_labels),
        "FMS": fowlkes_mallows_score(true_labels, new_labels),
        "VMS": v_measure_score(true_labels, new_labels),
    }
    log.info(
        "Accuracy                      = %.4f\n"
        "Fowlkes-Mallows score         = %.4f\n"
        "V-measure score               = %.4f",
        metrics["ACC"], metrics["FMS"], metrics["VMS"],
    )
    return metrics


def _run_clustering_algorithm(algorithm_desc, embedding_data, true_labels):
    """Run a single clustering method and return cluster labels."""
    if algorithm_desc["name"] == "LOO-KNN":
        return _loo_knn(embedding_data, true_labels)
    algo = algorithm_desc["class"](**algorithm_desc.get("params", {}))
    return algo.fit_predict(embedding_data)


_ALL_CLUSTERING_ALGORITHMS = [
    {"key": "knn",     "name": "LOO-KNN", "class": _loo_knn},
    {"key": "kmeans",  "name": "K-Means", "class": cluster.KMeans, "params": {"n_clusters": 10}},
    {"key": "hdbscan", "name": "HDBSCAN", "class": hdbscan.HDBSCAN, "params": {"min_cluster_size": 60}},
]


def _perform_and_evaluate_clustering(embedding_data, true_labels, embedding_name,
                                     algorithms=None, has_labels=True):
    """Run selected clustering methods, evaluate and log results.

    Returns list of dicts, one per algorithm, with all computed metrics.

    When has_labels=False, LOO-kNN is skipped (it's classification, not clustering),
    and only purely unsupervised metrics are reported.
    """
    if algorithms is None:
        algorithms = _ALL_CLUSTERING_ALGORITHMS

    results = []

    for algo_desc in algorithms:
        # LOO-kNN requires true labels — skip when unavailable
        if algo_desc["key"] == "knn" and not has_labels:
            log.info("Skipping LOO-kNN on %s (no labels available)", embedding_name)
            continue

        new_labels = _run_clustering_algorithm(algo_desc, embedding_data, true_labels)

        row = {"embedding": embedding_name, "method": algo_desc["key"], "pct_clustered": None}

        if algo_desc["name"] == "HDBSCAN":
            clustered_mask = new_labels >= 0
            if clustered_mask.any():
                unsup = _evaluate_and_log_unsupervised(
                    embedding_data[clustered_mask], true_labels[clustered_mask],
                    new_labels[clustered_mask], f"{algo_desc['name']} on {embedding_name}",
                    has_labels=has_labels,
                )
                row.update(unsup)
                if has_labels:
                    new_labels = _heuristic_labeling(true_labels, new_labels)
                    sup = _evaluate_and_log_supervised(
                        embedding_data[clustered_mask], true_labels[clustered_mask],
                        new_labels[clustered_mask],
                    )
                    row.update(sup)
            pct = np.sum(clustered_mask) / len(true_labels) * 100
            row["pct_clustered"] = round(pct, 1)
            log.info("HDBSCAN clustered             = %.1f%%", pct)
        else:
            unsup = _evaluate_and_log_unsupervised(
                embedding_data, true_labels, new_labels,
                f"{algo_desc['name']} on {embedding_name}",
                has_labels=has_labels,
            )
            row.update(unsup)
            if has_labels:
                new_labels = _heuristic_labeling(true_labels, new_labels)
                sup = _evaluate_and_log_supervised(embedding_data, true_labels, new_labels)
                row.update(sup)

        results.append(row)

    return results


_METRIC_COLUMNS = ["embedding", "method", "ARI", "AMI", "SS", "DBI", "CHI", "ACC", "FMS", "VMS", "pct_clustered"]


def save_metrics_csv(results: list[dict], output_path: str | Path) -> None:
    """Save clustering metrics to CSV."""
    df = pd.DataFrame(results, columns=_METRIC_COLUMNS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.4f")
    log.info("Cluster metrics saved to %s", output_path)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def cluster_latent(
    args,
    model,
    data: torch.Tensor,
    label_batch: torch.Tensor,
    output_dir: str | Path = "results",
) -> list[dict]:
    """
    Evaluate the latent space of a VAE on a batch of data.

    Runs clustering on the raw latent space, t-SNE, and UMAP embeddings.
    Optionally computes reconstruction probability scatter plots per class
    (only for single-level models with binary input).

    Parameters
    ----------
    args : argparse.Namespace
        Model/training configuration.
    model : nn.Module
        VAE model in eval mode.
    data : Tensor (N, D)
        Input samples.
    label_batch : Tensor (N,)
        Ground-truth labels.
    output_dir : str or Path
        Output folder for plots and metrics.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------- #
    # 0. Select methods and embeddings
    selected_methods = set(getattr(args, 'cluster_methods', ['knn', 'kmeans', 'hdbscan']))
    selected_embeddings = set(getattr(args, 'cluster_embeddings', ['raw', 'tsne', 'umap']))
    active_algorithms = [a for a in _ALL_CLUSTERING_ALGORITHMS if a['key'] in selected_methods]

    if not active_algorithms:
        log.warning("No clustering methods selected — skipping.")
        return []

    # ----------------------------------------------------- #
    # 1. Obtain latent representations
    with torch.no_grad():
        z_mean, z_logvar = model.q_z(data)
    latents = z_mean.detach().cpu().numpy()
    true_labels = label_batch.detach().cpu().numpy()

    # Detect if meaningful labels are available
    has_labels = len(np.unique(true_labels)) > 1
    if not has_labels:
        log.info("No meaningful labels detected — skipping supervised metrics "
                 "and LOO-kNN (classification). Only unsupervised metrics will be reported.")

    all_results = []

    # ----------------------------------------------------- #
    # 2. Raw latent space clustering
    if 'raw' in selected_embeddings:
        all_results.extend(
            _perform_and_evaluate_clustering(latents, true_labels, "raw",
                                             active_algorithms, has_labels=has_labels)
        )

    # ----------------------------------------------------- #
    # 3. t-SNE embedding
    umap_emb = None
    if 'tsne' in selected_embeddings:
        log.info("Computing t-SNE embedding...")
        tsne_emb = TSNE(
            n_components=2, random_state=42, perplexity=40,
        ).fit_transform(latents)

        all_results.extend(
            _perform_and_evaluate_clustering(tsne_emb, true_labels, "tsne",
                                             active_algorithms, has_labels=has_labels)
        )
        _scatter_plot(tsne_emb, true_labels, out_dir / "clustering_tsne.svg")

    # ----------------------------------------------------- #
    # 4. UMAP embedding
    if 'umap' in selected_embeddings:
        log.info("Computing UMAP embedding...")
        umap_emb = umap.UMAP(
            n_neighbors=30, min_dist=0.0, n_components=2, random_state=42,
        ).fit_transform(latents)

        all_results.extend(
            _perform_and_evaluate_clustering(umap_emb, true_labels, "umap",
                                             active_algorithms, has_labels=has_labels)
        )
        _scatter_plot(umap_emb, true_labels, out_dir / "clustering_umap.svg")

    if all_results:
        save_metrics_csv(all_results, out_dir / "cluster_metrics.csv")

    # ----------------------------------------------------- #
    # 5. Reconstruction probability scatter plots (per class)
    #    Uses the decoder to assess per-sample reconstruction quality,
    #    then visualizes best/worst samples and UMAP heatmaps.
    #    Only supported for single-level models (VAE, IWAE).
    if hasattr(model, 'q_z1'):
        log.info("Skipping reconstruction probability (hierarchical model)")
        return all_results

    if getattr(getattr(model, 'args', None), 'K', 1) > 1:
        log.info("Skipping reconstruction probability (IWAE K>1: p_x returns K*batch samples)")
        return all_results

    if args.input_type != 'binary':
        log.info("Skipping reconstruction probability (only supported for binary input)")
        return all_results

    if not has_labels:
        log.info("Skipping reconstruction probability (no labels for per-class analysis)")
        return all_results

    digit_folder = out_dir / "digits_recon_prob"
    digit_folder.mkdir(parents=True, exist_ok=True)
    num_classes = len(np.unique(true_labels))

    with torch.no_grad():
        for cls in range(num_classes):
            mask = (label_batch == cls)
            if mask.sum() < 10:
                continue
            data_cls = data[mask]
            z_mean_cls = z_mean[mask]
            z_logvar_cls = z_logvar[mask]

            # resample from posterior and decode
            z = model.reparameterize(z_mean_cls, z_logvar_cls)
            x_mean, x_logvar = model.p_x(z)

            # reconstruction log-probability per sample (handles all input types)
            recon_log_prob = model.reconstruction_loss(data_cls, x_mean, x_logvar)
            recon_scores = recon_log_prob.detach().cpu().numpy()

            # best and worst reconstructions
            n_show = min(9, len(recon_scores))
            top_idx = np.argpartition(-recon_scores, n_show)[:n_show]
            bottom_idx = np.argpartition(recon_scores, n_show)[:n_show]

            cls_str = f"class_{cls}"
            plot_images(
                args, data_cls[top_idx].detach().cpu().numpy(),
                str(digit_folder) + '/', cls_str + '_best', size_x=3, size_y=3,
            )
            plot_images(
                args, data_cls[bottom_idx].detach().cpu().numpy(),
                str(digit_folder) + '/', cls_str + '_worst', size_x=3, size_y=3,
            )

            # UMAP heatmap: how well does the best reconstruction match all samples?
            best_idx = int(recon_scores.argmax())
            N = data.size(0)
            best_xmean_exp = x_mean[best_idx].unsqueeze(0).expand(N, -1)
            # x_logvar may be: scalar 0., shape (1, D), or (batch, D)
            if isinstance(x_logvar, (int, float)):
                best_logvar_exp = x_logvar
            elif x_logvar.size(0) == 1:
                best_logvar_exp = x_logvar.expand(N, -1)
            else:
                best_logvar_exp = x_logvar[best_idx].unsqueeze(0).expand(N, -1)
            recon_all = model.reconstruction_loss(data, best_xmean_exp, best_logvar_exp)
            recon_all = recon_all.detach().cpu().numpy()

            if umap_emb is not None:
                plot_path = digit_folder / f"class_{cls}.svg"
                _scatter_plot(umap_emb, recon_all.astype(np.float64), plot_path, cmap='YlOrBr')

    log.info("Clustering analysis complete. Results saved to %s", out_dir)
    return all_results
