#!/usr/bin/env python3
"""Post-training analysis entry point.

Loads a pretrained model from a timestamp directory and runs one or more
analyses: clustering, reconstruction visualization, generation, KNN,
classification, OOD scores, cyclic generation.

Usage:
    uv run python analyze.py --dir pretrained_models/.../timestamp/ --cluster --recon_viz
    uv run python analyze.py --dir pretrained_models/.../timestamp/ --cluster --generate --KNN
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

from utils.load_data import load_dataset
from utils.model_io import importing_model, load_model


# ======================================================================
# Argument parsing
# ======================================================================
parser = argparse.ArgumentParser(
    description='Post-training analysis for VAE models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--dir', type=str, required=True,
                    help='path to timestamp directory (must contain checkpoint_best.pth)')

# --- Analysis flags (combine any number) ---------------------------------
g = parser.add_argument_group('Analysis flags')
g.add_argument('--cluster', action='store_true', default=False,
               help='cluster analysis (LOO-kNN, K-Means, HDBSCAN)')
g.add_argument('--cluster_methods', nargs='*',
               default=['knn', 'kmeans', 'hdbscan'],
               help='clustering methods to run')
g.add_argument('--cluster_embeddings', nargs='*',
               default=['raw', 'tsne', 'umap'],
               help='embedding spaces for clustering')
g.add_argument('--recon_viz', action='store_true', default=False,
               help='plot best/worst reconstructions per class by log p(x|z)')
g.add_argument('--generate', action='store_true', default=False,
               help='generate samples and reconstruction images')
g.add_argument('--KNN', action='store_true', default=False,
               help='KNN classification on latent space')
g.add_argument('--classify', action='store_true', default=False,
               help='train classifier on latent space')
g.add_argument('--ood_scores', action='store_true', default=False,
               help='OOD likelihood ratio scores')
g.add_argument('--ood_dataset', type=str, default=None,
               help='dataset name for OOD comparison')
g.add_argument('--cyclic_generation', action='store_true', default=False,
               help='cyclic generation through latent space')
g.add_argument('--export_latents', action='store_true', default=False,
               help='encode all input rows → z_mean.npy (preserves original row order)')
g.add_argument('--parquet_override', type=str, default=None,
               help='override config.parquet_path for --export_latents; encodes '
                    'an arbitrary parquet through the trained model (e.g. a full '
                    'cohort including groups the model was not trained on)')
g.add_argument('--out_z', type=str, default=None,
               help='override output path for --export_latents (default: '
                    '<model_dir>/z_mean.npy; use this to avoid overwriting the '
                    'training-set latents)')

# --- Settings -------------------------------------------------------------
g = parser.add_argument_group('Settings')
g.add_argument('--training_set_size', type=int, default=50000)
g.add_argument('--classification_dir', type=str, default='classification_report')
g.add_argument('--batch_size', type=int, default=100)
g.add_argument('--no-cuda', action='store_true', help='disable CUDA')
g.add_argument('--seed', type=int, default=42)

# --- Classifier hyperparameters (used by --classify) ----------------------
cl = parser.add_argument_group('Classifier (--classify)')
cl.add_argument('--classify_hidden_units', type=int, default=1024,
                help='Hidden layer size for the latent-space classifier')
cl.add_argument('--classify_lr', type=float, default=0.1,
                help='SGD learning rate for classifier training')
cl.add_argument('--classify_epochs', type=int, default=100,
                help='Training epochs for the classifier')
cl.add_argument('--classify_lambda', type=float, default=0.4,
                help='Weight for real vs augmented data loss (0=all augmented, 1=all real)')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# ======================================================================
# Model loading
# ======================================================================
def _load_pretrained_models(model_dir):
    """Load a pretrained model, its config, and data loaders from a timestamp directory.

    Returns (model, config, train_loader, val_loader, test_loader).
    """
    # Extract model_name from parent folder convention: ..._model_name=vae/timestamp/
    parent_folder = os.path.basename(os.path.dirname(model_dir.rstrip('/')))
    model_name_start = parent_folder.find('model_name=')
    if model_name_start == -1:
        print(f"Error: cannot extract model_name from parent folder '{parent_folder}'")
        print("Expected folder format: ..._model_name=<name>/timestamp/")
        sys.exit(1)
    model_name = parent_folder[model_name_start + len('model_name='):]

    config = torch.load(model_dir + model_name + '.config', weights_only=False)
    config.device = args.device
    VAE = importing_model(config)
    model = VAE(config)
    model.to(args.device)
    train_loader, val_loader, test_loader, config = load_dataset(
        config, training_num=args.training_set_size, no_binarization=True)
    load_model(model_dir + 'checkpoint_best.pth', model)
    model.eval()
    return model, config, train_loader, val_loader, test_loader


# ======================================================================
# Main
# ======================================================================
def main():
    directory = args.dir if args.dir.endswith('/') else args.dir + '/'

    # Validate path
    checkpoint_path = os.path.join(directory, 'checkpoint_best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        print("--dir must point to a timestamp directory containing checkpoint_best.pth")
        sys.exit(1)

    # Check at least one analysis flag is set
    analysis_flags = ['cluster', 'recon_viz', 'generate', 'KNN', 'classify',
                      'ood_scores', 'cyclic_generation', 'export_latents']
    if not any(getattr(args, f) for f in analysis_flags):
        print("Error: no analysis flag specified. Use one or more of:")
        print("  --cluster --recon_viz --generate --KNN --classify --ood_scores --cyclic_generation --export_latents")
        sys.exit(1)

    # Load model once
    print(f"Loading model from {directory}...")
    model, config, train_loader, val_loader, test_loader = _load_pretrained_models(directory)
    args.input_type = config.input_type
    args.input_size = config.input_size

    # --- Extract test data (needed by cluster, KNN) ---
    def _extract_test_tensors():
        all_data, all_labels = [], []
        for batch in test_loader:
            all_data.append(batch[0])
            all_labels.append(batch[-1])
        data = torch.cat(all_data).to(args.device)
        labels = torch.cat(all_labels).squeeze().to(args.device)
        return data, labels

    # --- Dispatch analyses sequentially ---

    if args.cluster:
        from utils.clustering import cluster_latent
        test_data, test_target = _extract_test_tensors()
        with torch.no_grad():
            results = cluster_latent(args, model, test_data, test_target, directory)
        print(f"Cluster analysis: {len(results)} metric rows -> {directory}cluster_metrics.csv")

    if args.recon_viz:
        from utils.visual_recon import plot_reconstruction_likelihood
        plot_reconstruction_likelihood(args, model, test_loader, directory)
        print(f"Reconstruction visualization saved to {directory}recon_likelihood/")

    if args.generate:
        with torch.no_grad():
            exemplars_n = 50
            selected_indices = torch.randint(
                low=0, high=config.training_set_size, size=(exemplars_n,))
            reference_images, indices, labels = train_loader.dataset[selected_indices]
            per_exemplar = 11
            generated = model.reference_based_generation_x(
                N=per_exemplar, reference_image=reference_images)
            if len(config.input_size) == 2:
                # TimeSeriesVAE: generated is (exemplars*N, D) flat — plot as line charts
                from utils.plot_timeseries import plot_ts_generation
                os.makedirs(directory + 'generated', exist_ok=True)
                seq_len = config.input_size[1]
                plot_ts_generation(
                    generated.cpu().numpy(), directory + 'generated/', 'generated',
                    seq_len=seq_len, n_samples=min(6, len(generated)),
                    plot_timesteps=getattr(config, 'plot_timesteps', None))
            else:
                from utils.plot_images import generate_fancy_grid
                generated = generated.reshape(-1, per_exemplar, *config.input_size)
                if config.use_logit:
                    reference_images = model.logit_inverse(reference_images)
                generate_fancy_grid(config, directory, reference_images, generated)
        print(f"Generation complete -> {directory}generated/")

    if args.KNN:
        from utils.knn_on_latent import report_knn_on_latent
        knn_dictionary = {'3': [], '5': [], '7': [], '10': [], '20': [],
                          '50': [], '75': [], '100': [], '200': []}
        with torch.no_grad():
            report_knn_on_latent(train_loader, val_loader, test_loader, model,
                                 directory, knn_dictionary, args, val=False)

    if args.classify:
        from utils.classify_data import classify_data
        # Map analyze.py arg names to classify_data.py expected names
        config.hidden_units = args.classify_hidden_units
        config.lr = args.classify_lr
        config.epochs = args.classify_epochs
        config.hyper_lambda = args.classify_lambda
        config.classification_dir = directory + 'classification/'
        test_acc, val_acc = classify_data(
            train_loader, val_loader, test_loader,
            config.classification_dir, config, model)
        print(f"Test accuracy: {test_acc:.2f}, Val accuracy: {val_acc:.2f}")

    if args.ood_scores:
        from utils.ood_scores import decompose_elbo, save_ood_scores, plot_ood_histograms
        from utils.evaluation import load_all_pseudo_input
        from models.AbsHModel import BaseHModel

        if not isinstance(model, BaseHModel):
            print('Error: OOD score decomposition requires a hierarchical model.')
            sys.exit(1)

        original_K = getattr(config, 'K', 1)
        config.K = 1

        with torch.no_grad():
            exemplars_embedding = load_all_pseudo_input(config, model, train_loader.dataset)
            id_scores = decompose_elbo(model, test_loader, args.device, exemplars_embedding)

            ood_scores = None
            if args.ood_dataset:
                ood_config = copy.deepcopy(config)
                ood_config.dataset_name = args.ood_dataset
                _, _, ood_test_loader, ood_config = load_dataset(
                    ood_config, training_num=args.training_set_size, no_binarization=True)
                if list(ood_config.input_size) != list(config.input_size):
                    print(f'Error: OOD input_size {ood_config.input_size} != model {config.input_size}')
                    sys.exit(1)
                ood_scores = decompose_elbo(model, ood_test_loader, args.device,
                                            exemplars_embedding)

        save_ood_scores(id_scores, ood_scores, directory + 'ood_scores.csv')
        print(f'OOD scores saved to {directory}ood_scores.csv')
        if ood_scores is not None:
            plot_ood_histograms(id_scores, ood_scores, directory + 'ood_histograms.png')

        config.K = original_K

    if args.cyclic_generation:
        from utils.plot_images import plot_images
        with torch.no_grad():
            z_start = model.reparameterize(
                *model.q_z(next(iter(test_loader))[0][:1].to(args.device)))
            z_end = model.reparameterize(
                *model.q_z(next(iter(test_loader))[0][1:2].to(args.device)))
            steps = 10
            alphas = torch.linspace(0, 1, steps).to(args.device)
            z_interp = z_start * (1 - alphas.unsqueeze(1)) + z_end * alphas.unsqueeze(1)
            x_mean, _ = model.p_x(z_interp)
            plot_images(config, x_mean.detach().cpu().numpy(), directory,
                        'cyclic_generation', size_x=2, size_y=5)
        print(f"Cyclic generation saved to {directory}cyclic_generation.png")

    if args.export_latents:
        import re
        import pandas as pd
        csv_path = getattr(config, 'csv_path', None)
        # Parquet source: explicit --parquet_override wins over the training
        # config's parquet_path. This lets us encode an arbitrary ROI set
        # (e.g. full cohort) through a model that was trained on a subset.
        parquet_path = args.parquet_override or getattr(config, 'parquet_path', None)
        if not csv_path and not parquet_path:
            print("Error: --export_latents requires a model trained with --csv_path or --parquet_path, "
                  "or --parquet_override pointing to a parquet file")
            sys.exit(1)

        if parquet_path:
            # parquet_timeseries: sample columns match 't_<digits>'; rest is meta
            df = pd.read_parquet(parquet_path)
            t_pat = re.compile(r"^t_\d+$")
            t_cols = sorted(c for c in df.columns if t_pat.match(c))
            x_all = df[t_cols].to_numpy(dtype=np.float32)
            print(f"Encoding from parquet: {parquet_path} (shape {x_all.shape})")
        else:
            n_meta = getattr(config, 'csv_meta_cols', 2)
            df = pd.read_csv(csv_path, sep='\t')
            x_all = df.iloc[:, n_meta:].values.astype(np.float32)
            print(f"Encoding from csv: {csv_path} (shape {x_all.shape})")
        x_min, x_max = x_all.min(), x_all.max()
        x_all = (x_all - x_min) / (x_max - x_min + 1e-7)

        # Encode all rows in original order, matching training normalisation
        parts = []
        tensor = torch.tensor(x_all, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, len(tensor), args.batch_size):
                batch = tensor[i:i + args.batch_size].to(args.device)
                z_mean, _ = model.q_z(batch)
                parts.append(z_mean.cpu().numpy())
        z_mean = np.concatenate(parts, axis=0)  # (N, z_dim)

        out_path = args.out_z or os.path.join(directory, 'z_mean.npy')
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.save(out_path, z_mean)
        print(f"LATENTS_PATH: {out_path}")
        print(f"Exported z_mean: shape {z_mean.shape} → {out_path}")

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
