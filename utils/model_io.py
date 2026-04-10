import torch
import os

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
# Model registry — maps model_name string to the corresponding model class
# ======================================================================================================================
def importing_model(args):
    """Map args.model_name string to the corresponding model class.

    Uses lazy imports so only the requested model module is loaded.
    Returns the class itself (not an instance) — caller does ``model = VAE(args)``.
    """
    if args.model_name == 'vae':
        from models.VAE import VAE
    elif args.model_name == 'iwae':
        from models.IWAE import IWAE
        return IWAE
    elif args.model_name == 'iwae_2level':
        from models.IWAE_2level import IWAE_2level
        return IWAE_2level
    elif args.model_name == 'timeseries_vae':
        from models.TimeSeriesVAE import TimeSeriesVAE
        return TimeSeriesVAE
    elif args.model_name == 'hvae_2level':
        from models.HVAE_2level import VAE
    elif args.model_name == 'convhvae_2level':
        from models.convHVAE_2level import VAE
    elif args.model_name == 'pixelcnn':
        from models.PixelCNN import VAE
    elif args.model_name == 'convvae':
        from models.ConvVAE import ConvVAE
        return ConvVAE
    else:
        raise Exception('Wrong name of the model!')
    return VAE


# ======================================================================================================================
# Checkpoint I/O — atomic save and state_dict restore
# ======================================================================================================================
def save_model(save_path, load_path, content):
    """Atomic checkpoint save: write to temp path, then rename."""
    torch.save(content, save_path)
    os.rename(save_path, load_path)


def load_model(load_path, model, optimizer=None):
    """Restore model (and optionally optimizer) state from a checkpoint."""
    checkpoint = torch.load(load_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint
