"""
Unit tests for utils.load_model.

Tests cover:
  - Restoring model weights from a checkpoint
  - Restoring args/config from a saved Namespace
  - Error handling for missing dirs, bad folder names, missing files
"""

import argparse
import os

import pytest
import torch
import torch.optim as optim

from models.VAE import VAE
from utils.load_model import load_model


# ======================================================================================================================
# Helper: build a minimal args Namespace
# ======================================================================================================================
def make_args(**overrides):
    defaults = dict(
        model_name='vae',
        prior='standard',
        input_type='binary',
        input_size=[1, 28, 28],
        hidden_size=300,
        z1_size=40,
        z2_size=40,
        activation=None,
        no_attention=False,
        same_variational_var=False,
        use_logit=False,
        number_components=500,
        pseudoinputs_mean=-0.05,
        pseudoinputs_std=0.01,
        use_training_data_init=False,
        dataset_name='dynamic_mnist',
        training_set_size=60000,
        dynamic_binarization=False,
        lr=5e-4,
        device=torch.device('cpu'),
        cuda=False,
        use_whole_train=False,
        approximate_prior=False,
        approximate_k=10,
        continuous=False,
        lambd=1e-4,
        bottleneck=6,
        K=1,
        IW=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_parser():
    """Build a parser matching run.py's argument definitions."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--early_stopping_epochs', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--h_size', type=int, default=300)
    parser.add_argument('--z1_size', type=int, default=40)
    parser.add_argument('--z2_size', type=int, default=40)
    parser.add_argument('--input_size', type=int, default=[1, 28, 28])
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--same_variational_var', action='store_true')
    parser.add_argument('--number_components', type=int, default=500)
    parser.add_argument('--pseudoinputs_mean', type=float, default=-0.05)
    parser.add_argument('--pseudoinputs_std', type=float, default=0.01)
    parser.add_argument('--use_training_data_init', action='store_true')
    parser.add_argument('--model_name', type=str, default='vae')
    parser.add_argument('--prior', type=str, default='standard')
    parser.add_argument('--input_type', type=str, default='gray')
    parser.add_argument('--K', type=int, nargs='?', const=1, default=1)
    parser.add_argument('--IW', action='store_true')
    parser.add_argument('--ll', action='store_true', default=None, dest='ll')
    parser.add_argument('--no_ll', action='store_false', dest='ll')
    parser.add_argument('--S', type=int, default=5000)
    parser.add_argument('--dataset_name', type=str, default='dynamic_mnist')
    parser.add_argument('--dynamic_binarization', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--use_logit', action='store_true')
    parser.add_argument('--lambd', type=float, default=1e-4)
    return parser


def create_fake_checkpoint(tmp_path, config_args):
    """Create a fake pretrained_models directory with config and checkpoint.

    Returns (run_dir, original_state_dict).
    """
    run_dir = tmp_path / "test_standard_model_name=vae" / "run_1"
    run_dir.mkdir(parents=True)

    # save config
    torch.save(config_args, run_dir / "vae.config")

    # create model, save its state_dict
    model = VAE(config_args)
    optimizer = optim.Adam(model.parameters(), lr=config_args.lr)
    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    checkpoint = {
        'epoch': 10,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': 88.0,
        'e': 0,
    }
    torch.save(checkpoint, run_dir / "checkpoint_best.pth")

    return str(run_dir), original_state_dict


# ======================================================================================================================
# Tests: successful loading
# ======================================================================================================================
class TestLoadModelSuccess:
    def test_restores_weights(self, tmp_path, monkeypatch):
        """Loaded model weights must match the saved checkpoint exactly."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        config = make_args()
        run_dir, original_sd = create_fake_checkpoint(tmp_path, config)

        args = make_args(dir=run_dir)
        parser = make_parser()
        model, optimizer, restored_args, snap_dir = load_model(args, parser)

        for key in original_sd:
            assert torch.equal(model.state_dict()[key], original_sd[key]), \
                f"Mismatch in parameter: {key}"

    def test_restores_args(self, tmp_path, monkeypatch):
        """Restored args must preserve key config values."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        config = make_args(z1_size=20, prior='standard', input_type='binary')
        run_dir, _ = create_fake_checkpoint(tmp_path, config)

        args = make_args(dir=run_dir)
        parser = make_parser()
        _, _, restored_args, _ = load_model(args, parser)

        assert restored_args.model_name == 'vae'
        assert restored_args.z1_size == 20
        assert restored_args.prior == 'standard'
        assert restored_args.input_type == 'binary'

    def test_restores_optimizer_state(self, tmp_path, monkeypatch):
        """Optimizer state dict must be restored from checkpoint."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        config = make_args()
        run_dir, _ = create_fake_checkpoint(tmp_path, config)

        args = make_args(dir=run_dir)
        parser = make_parser()
        _, optimizer, _, _ = load_model(args, parser)

        assert isinstance(optimizer, torch.optim.Optimizer)
        # optimizer should have state loaded (param_groups populated)
        assert len(optimizer.param_groups) > 0

    def test_snap_dir_has_trailing_slash(self, tmp_path, monkeypatch):
        """snap_dir must end with '/' for path concatenation."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        config = make_args()
        run_dir, _ = create_fake_checkpoint(tmp_path, config)

        args = make_args(dir=run_dir)
        parser = make_parser()
        _, _, _, snap_dir = load_model(args, parser)

        assert snap_dir.endswith('/')


# ======================================================================================================================
# Tests: error handling
# ======================================================================================================================
class TestLoadModelErrors:
    def test_missing_dir_raises(self, monkeypatch):
        """args.dir = None should raise ValueError."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        args = make_args(dir=None)
        parser = make_parser()
        with pytest.raises(ValueError, match="No --dir provided"):
            load_model(args, parser)

    def test_nonexistent_dir_raises(self, tmp_path, monkeypatch):
        """Pointing to a directory that doesn't exist should raise FileNotFoundError."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        args = make_args(dir=str(tmp_path / "nonexistent" / "path"))
        parser = make_parser()
        with pytest.raises(FileNotFoundError, match="Cannot find run directory"):
            load_model(args, parser)

    def test_bad_folder_name_raises(self, tmp_path, monkeypatch):
        """Folder without 'model_name=' in parent should raise ValueError."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        bad_dir = tmp_path / "no_model_marker" / "run_1"
        bad_dir.mkdir(parents=True)
        args = make_args(dir=str(bad_dir))
        parser = make_parser()
        with pytest.raises(ValueError, match="Cannot parse model_name"):
            load_model(args, parser)

    def test_missing_config_raises(self, tmp_path, monkeypatch):
        """Directory exists but no .config file should raise FileNotFoundError."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        run_dir = tmp_path / "test_model_name=vae" / "run_1"
        run_dir.mkdir(parents=True)
        # create checkpoint but no config
        torch.save({'state_dict': {}, 'optimizer': {}}, run_dir / "checkpoint_best.pth")
        args = make_args(dir=str(run_dir))
        parser = make_parser()
        with pytest.raises(FileNotFoundError, match="Cannot find config file"):
            load_model(args, parser)

    def test_missing_checkpoint_raises(self, tmp_path, monkeypatch):
        """Directory with config but no checkpoint should raise FileNotFoundError."""
        monkeypatch.setattr('sys.argv', ['run.py'])
        run_dir = tmp_path / "test_model_name=vae" / "run_1"
        run_dir.mkdir(parents=True)
        config = make_args()
        torch.save(config, run_dir / "vae.config")
        args = make_args(dir=str(run_dir))
        parser = make_parser()
        with pytest.raises(FileNotFoundError, match="Cannot find checkpoint file"):
            load_model(args, parser)
