"""
Instantiate a new VAE model and prepare its output directory.

Directory structure (same convention as analysis.py and load_model.py):

    pretrained_models/
      {description}_model_name={model_name}/   # outer folder — encodes model_name
        {timestamp}/                           # inner folder — one per training run
          {model_name}.config
          checkpoint_best.pth
          ...

The ``model_name=`` substring in the outer folder is parsed by analysis.py
and load_model.py to determine which model class to instantiate on load.
"""

import datetime
import os

from utils.optimizer import AdamNormGrad
from utils.model_io import importing_model

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
# Model factory
# ======================================================================================================================
def create_model(args):
    """
    Build a fresh VAE model and set up the output directory.

    As a side effect, sets ``args.timestamp`` (runtime field) to the
    current datetime string used for the run directory name.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    model : nn.Module
        The newly created model, already moved to ``args.device``.
    optimizer : AdamNormGrad
        Adam optimizer with L2-normalized gradients.
    args : argparse.Namespace
        Arguments augmented with ``args.timestamp`` runtime field.
    snap_dir : str
        Path to the run directory where checkpoints will be saved.
    """
    # outer folder encodes dataset, prior, and model_name so that
    # analysis.py can parse the model class from the directory name
    outer_folder = (
        f"{args.dataset_name}_{args.prior}"
        f"_model_name={args.model_name}"
    )

    # runtime field: run_id identifies this specific training run
    # When running under Slurm, use job/task IDs for reproducible directory names;
    # otherwise fall back to a timestamp.
    if getattr(args, 'slurm_job_id', '') != '':
        run_id = f"slurm_{args.slurm_job_id}"
        if getattr(args, 'slurm_task_id', '') != '':
            run_id += f"_{args.slurm_task_id}"
    else:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    args.timestamp = run_id

    snap_dir = os.path.join('pretrained_models', outer_folder, run_id, '')
    os.makedirs(snap_dir, exist_ok=True)

    # importing_model() maps args.model_name → model class
    VAE = importing_model(args)

    print('Creating model...')
    model = VAE(args)
    model.to(args.device)

    # --- Optimizer ---
    # AdamNormGrad normalizes gradients by L2 norm before applying Adam.
    # This is the optimizer used in the VampPrior and Exemplar VAE papers:
    #   https://arxiv.org/abs/1705.07120 (Tomczak & Welling, 2018)
    #   https://arxiv.org/abs/2004.04795 (Aneja et al., 2021)
    #
    # --- Alternative: standard Adam (uncomment to replace) ----------------
    #   import torch.optim as optim
    #   optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # ----------------------------------------------------------------------
    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)

    print(args)

    return model, optimizer, args, snap_dir
