#!/usr/bin/env python3
"""
Main entry point for running VAE experiments.

This script parses a set of command‑line arguments,
prepares the dataset, builds or loads the requested model,
and finally runs the training / evaluation loop.
"""

# ======================================================================================================================
# Imports
# ======================================================================================================================

import argparse

import torch

from utils.load_data import load_dataset
from utils.logger_config import setup_logging
from utils.load_model import load_model
from utils.create_model import create_model

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #

# ======================================================================================================================
# Argument parsing
# ======================================================================================================================

parser = argparse.ArgumentParser(
    description='Variational Autoencoder',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# --- Loading saved models / config ------------------------------------------
g = parser.add_argument_group('Loading saved models / config')
g.add_argument('--dir',    type=str, default=None, help='path to a run directory to load')
g.add_argument('--config', '-c', type=str, default=None,
               help='YAML config file (CLI args override YAML values)')

# --- Optimisation ------------------------------------------------------------
g = parser.add_argument_group('Optimisation')
g.add_argument('--batch_size',            type=int,   default=100,  help='training batch size')
g.add_argument('--test_batch_size',       type=int,   default=100,  help='test batch size')
g.add_argument('--epochs',                type=int,   default=2000, help='number of training epochs')
g.add_argument('--lr',                    type=float, default=5e-4, help='learning rate')
g.add_argument('--early_stopping_epochs', type=int,   default=50,   help='patience for early stopping')
g.add_argument('--warmup',                type=int,   default=100,  help='warm-up epochs')

# --- Logging -----------------------------------------------------------------
g = parser.add_argument_group('Logging')

# --- Device / seed -----------------------------------------------------------
g = parser.add_argument_group('Device / seed')
g.add_argument('--no-cuda', action='store_true', help='disable CUDA')
g.add_argument('--seed',    type=int, default=42, help='random seed')

# --- Architecture ------------------------------------------------------------
g = parser.add_argument_group('Architecture')
g.add_argument('--h_size',     type=int, default=300,        help='hidden layer size')
g.add_argument('--z1_size',    type=int, default=40,         help='latent dim z1')
g.add_argument('--z2_size',    type=int, default=40,         help='latent dim z2')
g.add_argument('--input_size', type=int, default=[1, 28, 28], help='input dimensions')
g.add_argument('--activation', type=str, default=None,       help='activation function')
g.add_argument('--no_attention',        action='store_true', help='disable attention in GatedDense layers')
g.add_argument('--same_variational_var', action='store_true', help='use scalar learned variance instead of per-dim')

# --- Adaptive latent size discovery ------------------------------------------
g = parser.add_argument_group('Adaptive latent size')
g.add_argument('--auto_z_size',        action='store_true',       help='enable adaptive latent dimension discovery')
g.add_argument('--auto_accept_z',      action='store_true',       help='auto-accept AU restart without interactive prompt (for non-interactive / CI runs)')
g.add_argument('--au_check_interval',  type=int,   default=5,     help='compute active units every N epochs')
g.add_argument('--au_stability_count', type=int,   default=5,     help='consecutive stable readings before prompting')
g.add_argument('--au_threshold',       type=float, default=0.01,  help='variance threshold for active dimension')

# --- VampPrior ---------------------------------------------------------------
g = parser.add_argument_group('VampPrior')
g.add_argument('--number_components',     type=int,   default=500,   help='number of pseudo-inputs')
g.add_argument('--pseudoinputs_mean',     type=float, default=-0.05, help='pseudo-input init mean')
g.add_argument('--pseudoinputs_std',      type=float, default=0.01,  help='pseudo-input init std')
g.add_argument('--use_training_data_init', action='store_true',      help='init pseudo-inputs from data')

# --- Model -------------------------------------------------------------------
g = parser.add_argument_group('Model')
g.add_argument('--model_name', type=str, default='vae',      choices=['vae', 'iwae', 'iwae_2level', 'hvae_2level', 'convhvae_2level', 'pixelhvae_2level', 'timeseries_vae', 'convvae'], help='model architecture')
g.add_argument('--prior',      type=str, default='standard',  choices=['standard', 'vampprior', 'exemplar_prior'], help='prior type')
g.add_argument('--input_type', type=str, default='gray',      choices=['binary', 'gray', 'continuous'], help='input data type')

# --- Sampling / importance weighting -----------------------------------------
g = parser.add_argument_group('Sampling')
g.add_argument('--K',  type=int, nargs='?', const=1, default=1, help='number of latent samples')
g.add_argument('--IW', action='store_true',                      help='enable importance weighting')

# --- Evaluation --------------------------------------------------------------
g = parser.add_argument_group('Evaluation')
g.add_argument('--ll', action='store_true', default=None, dest='ll',
               help='compute log-likelihood after training/loading (default: auto)')
g.add_argument('--no_ll', action='store_false', dest='ll',
               help='skip log-likelihood computation')
g.add_argument('--S',  type=int, default=5000, help='importance samples for log-likelihood estimation')
g.add_argument('--cluster', action='store_true', default=False,
               help='run clustering analysis on test set (LOO-kNN, K-Means, HDBSCAN, t-SNE, UMAP)')
g.add_argument('--cluster_methods', nargs='*',
               default=['knn', 'kmeans', 'hdbscan'],
               help='clustering methods to run (default: all)')
g.add_argument('--cluster_embeddings', nargs='*',
               default=['raw', 'tsne', 'umap'],
               help='embedding spaces for clustering (default: all)')
g.add_argument('--recon_viz', action='store_true', default=False,
               help='plot best/worst reconstructions per class after training')
g.add_argument('--generate', action='store_true', default=False,
               help='generate samples and reconstructions after training')

# --- Dataset -----------------------------------------------------------------
g = parser.add_argument_group('Dataset')
g.add_argument('--dataset_name', type=str, default='dynamic_mnist',
               choices=['static_mnist', 'dynamic_mnist', 'fashion_mnist',
                        'omniglot', 'caltech101silhouettes',
                        'histopathologyGray', 'freyfaces',
                        'svhn', 'cifar10',
                        'ecg5000', 'synthetic_timeseries',
                        'tabular_timeseries'],
               help='dataset to use')

# --- Time series --------------------------------------------------------------
g = parser.add_argument_group('Time series')
g.add_argument('--seq_len',    type=int, default=50,  help='sequence length (set automatically by dataset)')
g.add_argument('--feat_dim',   type=int, default=1,   help='features per timestep')
g.add_argument('--trend_poly', type=int, default=0,   help='polynomial degree for trend (0=disabled)')
g.add_argument('--use_residual', action='store_true', default=True,
               help='use ConvTranspose1d residual connection in decoder')
g.add_argument('--reconstruction_wt', type=float, default=3.0,
               help='reconstruction loss weight (TimeVAE default=3.0)')
g.add_argument('--reconstruction_dist', type=str, default='beta',
               choices=['beta', 'gaussian'],
               help='reconstruction distribution for time series (default: beta)')
g.add_argument('--plot_timesteps', type=int, default=None,
               help='max timesteps to show in plots (default: all)')
g.add_argument('--dynamic_binarization', action='store_true', help='enable dynamic binarization')
g.add_argument('--continuous',           action='store_true', help='treat data as continuous (gray) instead of binary')
g.add_argument('--use_logit',            action='store_true', help='apply logit preprocessing to continuous data')
g.add_argument('--lambd', type=float, default=1e-4,          help='lambda for logit transform (avoids log(0))')

# --- Tabular time-series (bring-your-own data: CSV / TSV / Parquet / NPY) ----
# See API.md ("Tabular time-series — bring your own data") for the full reference.
g = parser.add_argument_group('Tabular time-series')
g.add_argument('--ts_path', type=str, default=None,
               help='single-file path (csv/tsv/parquet/npy); auto-split into train/val/test '
                    'per --ts_split. Mutually exclusive with --ts_train_path et al.')
g.add_argument('--ts_train_path', type=str, default=None,
               help='pre-split train file (must be paired with --ts_val_path and --ts_test_path)')
g.add_argument('--ts_val_path', type=str, default=None,
               help='pre-split validation file')
g.add_argument('--ts_test_path', type=str, default=None,
               help='pre-split test file')
g.add_argument('--ts_format', type=str, default='auto',
               choices=['auto', 'csv', 'parquet', 'npy'],
               help='file format; auto-detected from extension when "auto"')
g.add_argument('--ts_csv_sep', type=str, default=',',
               help="separator for csv format; default ',' but auto-falls-back to '\\t' "
                    "for .tsv extension when sep is left at default")
g.add_argument('--ts_value_cols', type=str, default=None,
               help="sample-column selector. Regex if no comma (e.g. '^t_\\d+$'); "
                    "comma-separated explicit list if comma present (e.g. 'a,b,c'). "
                    "Required for csv/parquet; ignored for npy.")
g.add_argument('--ts_label_col', type=str, default=None,
               help='optional label column; pd.factorize maps strings to ints. '
                    'When None, all labels are 0 (anomaly-detection style). Ignored for npy.')
g.add_argument('--ts_split', type=str, default='0.8/0.1/0.1',
               help="train/val/test ratios as 'a/b/c' summing to 1.0; "
                    "ignored when pre-split paths are supplied")
g.add_argument('--ts_normalise', type=str, default='global_minmax',
               choices=['global_minmax', 'per_sample_minmax', 'zscore', 'none'],
               help='normalisation mode; statistics computed from train split only')

# --- Exemplar prior -----------------------------------------------------------
g = parser.add_argument_group('Exemplar prior')
g.add_argument('--no_mask',           action='store_true', default=False,
               help='disable leave-one-out masking in exemplar prior')
g.add_argument('--approximate_prior', action='store_true', default=False,
               help='use approximate nearest-neighbor exemplar set')
g.add_argument('--approximate_k',     type=int, default=10,
               help='k nearest neighbors for approximate prior')

# --- Slurm --------------------------------------------------------------------
g = parser.add_argument_group('Slurm')
g.add_argument('--slurm_job_id',  type=str, default='', help='Slurm job ID (for directory naming)')
g.add_argument('--slurm_task_id', type=str, default='', help='Slurm array task ID')

# ----------------------------------------------------------------------
# Parse the arguments
# ----------------------------------------------------------------------
# If --config is provided, load YAML and use its values as defaults.
# CLI args override YAML values. When --dir is also set, the saved
# model config overrides YAML (see load_model.py).
# Priority: CLI > saved model config > YAML > parser defaults.
import yaml

pre_args, _ = parser.parse_known_args()
if pre_args.config:
    with open(pre_args.config) as f:
        config = yaml.safe_load(f)
    flat = {}
    for section_value in config.values():
        if isinstance(section_value, dict):
            flat.update(section_value)
    parser.set_defaults(**flat)

args = parser.parse_args()

# ======================================================================
# Runtime fields — derived from CLI args, not from command line directly
# ======================================================================
args.cuda        = not args.no_cuda and torch.cuda.is_available()
args.device      = torch.device("cuda" if args.cuda else "cpu")
args.hidden_size = args.h_size   # alias used internally by models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# ======================================================================================================================
# Main experiment runner
# ======================================================================================================================
def run(args, kwargs):
    """
    Orchestrates a full VAE experiment.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command‑line arguments.
    kwargs : dict
        Keyword arguments passed to the data loader.
    """

    setup_logging()

    if args.dir:
        # --dir mode: restore config FIRST (includes dataset_name, input_size,
        # input_type, etc.), THEN load dataset with the correct settings.
        model, optimizer, args, snap_dir = load_model(args, parser)
        print('Loading data...')
        train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)
    else:
        # Training mode: load dataset first (sets input_size, input_type which
        # the model constructor needs), then create the model.
        print('Loading data...')
        train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)
        model, optimizer, args, snap_dir = create_model(args)

    # ======================================================================
    # Run the experiment
    # ======================================================================
    print('Performing experiment...')
    from utils.perform_experiment import experiment_vae

    experiment_vae(
        args,
        train_loader,
        val_loader,
        test_loader,
        model,
        optimizer,
        snap_dir,
        model_name=args.model_name
    )

    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    print('All done...')


# ======================================================================================================================
# Entry point guard
# ======================================================================================================================
if __name__ == "__main__":
    run(args, kwargs)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
