"""
Load a previously saved VAE model from a run directory.

Instead of loading an entire pickled model object, this module loads:
1. A config file (.config) — contains the model's hyperparameters/args
2. A checkpoint file (.pth) — contains the model's learned weights (state_dict)

This is safer and more portable than pickling the whole model.

Expected directory structure (same convention as analysis.py):

    pretrained_models/
      {description}_model_name={model_name}/   # outer folder
        {run_id}/                              # inner folder
          {model_name}.config
          checkpoint_best.pth
"""

import os

import torch

from utils.optimizer import AdamNormGrad
from utils.model_io import importing_model


def load_model(args, parser):
    """
    Load a saved VAE model and restore its configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Must include ``dir`` pointing to
        the full run directory path, e.g.
        ``pretrained_models/exemplar_prior_on_dynamic_mnist_model_name=vae/1``
    parser : argparse.ArgumentParser
        The argument parser, used to re-parse with restored defaults.

    Returns
    -------
    model : nn.Module
        The restored model with loaded weights.
    optimizer : AdamNormGrad
        An optimizer wrapping the model parameters (with restored state if available).
    args : argparse.Namespace
        Arguments after restoring saved defaults.
    snap_dir : str
        Path to the run directory.
    """
    if not args.dir:
        raise ValueError(
            "No --dir provided. "
            "Please supply the path to the run directory you want to restore, e.g. "
            "pretrained_models/exemplar_prior_on_dynamic_mnist_model_name=vae/1"
        )

    # --- Locate the run directory ---
    # --dir should point directly to the run folder, e.g.
    # "pretrained_models/exemplar_prior_on_dynamic_mnist_model_name=vae/1"
    snap_dir = args.dir
    if not snap_dir.endswith('/'):
        snap_dir += '/'

    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"Cannot find run directory: {snap_dir}")

    # --- Parse model_name from the parent folder ---
    # The outer folder (parent of the run folder) encodes model_name after "model_name=",
    # e.g. "exemplar_prior_on_dynamic_mnist_model_name=vae" → model_name = "vae"
    # This is the same convention used by analysis.py.
    parent_folder = os.path.basename(os.path.dirname(snap_dir.rstrip('/')))
    marker = 'model_name='
    marker_index = parent_folder.find(marker)
    if marker_index == -1:
        raise ValueError(
            f"Cannot parse model_name from folder '{parent_folder}'. "
            f"Expected folder name to contain 'model_name=<name>'."
        )
    model_name = parent_folder[marker_index + len(marker):]

    # --- Step 1: Load the config (saved hyperparameters / args) ---
    # The .config file is a small object saved with torch.save() that holds
    # all the settings used during training (e.g. latent dim, prior type, etc.)
    config_path = os.path.join(snap_dir, f"{model_name}.config")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Cannot find config file: {config_path}")

    config = torch.load(config_path, map_location=args.device, weights_only=False)

    # --- Step 2: Build the model architecture from the config ---
    # importing_model() looks at config.model_name (e.g. 'vae', 'hvae_2level')
    # and returns the corresponding model class.
    VAE = importing_model(config)

    # Now we create an empty model with the right architecture — no learned
    # weights yet, just the skeleton (layers, dimensions, etc.)
    config.device = args.device
    model = VAE(config)
    model.to(args.device)

    # --- Step 3: Load the checkpoint (learned weights) into the model ---
    # The .pth file is a dictionary containing:
    #   - 'state_dict': the model's learned parameters
    #   - 'optimizer':   the optimizer's state (momentum, etc.)
    #   - 'epoch', 'best_loss', etc.
    checkpoint_path = os.path.join(snap_dir, "checkpoint_best.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Cannot find checkpoint file: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # Apply the saved weights to our empty model skeleton
    model.load_state_dict(checkpoint['state_dict'])

    # --- Step 4: Restore the training args ---
    # We take the saved config values and set them as parser defaults,
    # then re-parse so that any CLI overrides still take effect.
    model_args = vars(config)
    parser.set_defaults(**model_args)
    args = parser.parse_args()

    # --- Step 5: Set up the optimizer ---
    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)

    # If the checkpoint contains a saved optimizer state, restore it
    # so training can resume smoothly from where it left off.
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, args, snap_dir
