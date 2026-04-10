"""End-to-end test for utils/perform_experiment.py.

Creates a real VAE model with tiny synthetic data and runs experiment_vae()
for 3 epochs to verify the full training loop produces expected outputs.
"""
import os
import shutil

import pytest
import torch
from types import SimpleNamespace

from models.VAE import VAE
from utils.optimizer import AdamNormGrad
from utils.perform_experiment import experiment_vae


@pytest.fixture
def tiny_setup(tmp_path):
    """Create a minimal VAE + data loaders + output dir for 3-epoch training."""
    args = SimpleNamespace(
        # --- model ---
        model_name='vae',
        prior='standard',
        input_type='binary',
        input_size=[1, 28, 28],
        z1_size=8,
        z2_size=8,
        h_size=32,
        hidden_size=32,
        K=1,
        IW=False,
        # --- training ---
        epochs=3,
        warmup=1,
        early_stopping_epochs=10,
        lr=1e-3,
        # --- evaluation ---
        ll=False,       # skip LL (too slow for test)
        S=10,
        cluster=False,
        recon_viz=False,
        generate=False,
        # --- data ---
        dataset_name='dynamic_mnist',
        training_set_size=100,
        dynamic_binarization=True,
        # --- device ---
        device=torch.device('cpu'),
        cuda=False,
        # --- misc ---
        dir=None,         # train mode (not load mode)
        auto_z_size=False,
        number_components=5,
        pseudoinputs_mean=-0.05,
        pseudoinputs_std=0.01,
        use_training_data_init=False,
        no_attention=False,
        same_variational_var=False,
        use_logit=False,
    )

    # Build model
    model = VAE(args)
    model.to(args.device)
    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)

    # Tiny synthetic data (100 train, 20 val, 20 test)
    n_train, n_val, n_test = 100, 20, 20
    input_dim = 784  # 1 * 28 * 28

    def make_train_loader(n, batch_size=20):
        """Train loader yields (data, indices, target) — 3-tuple format
        required by train_vae() for exemplar prior LOO masking."""
        data = torch.rand(n, input_dim)
        indices = torch.arange(n)
        labels = torch.randint(0, 10, (n,))
        dataset = torch.utils.data.TensorDataset(data, indices, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def make_eval_loader(n, batch_size=20):
        """Val/test loaders yield (data, target) — 2-tuple format."""
        data = torch.rand(n, input_dim)
        labels = torch.randint(0, 10, (n,))
        dataset = torch.utils.data.TensorDataset(data, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    train_loader = make_train_loader(n_train)
    val_loader = make_eval_loader(n_val)
    test_loader = make_eval_loader(n_test)

    # Output directory (evaluate_vae saves reconstruction images to reconstruction/ subdir)
    snap_dir = str(tmp_path) + '/'
    os.makedirs(snap_dir + 'reconstruction', exist_ok=True)

    return args, model, optimizer, train_loader, val_loader, test_loader, snap_dir


class TestExperimentVAE:
    def test_creates_checkpoint(self, tiny_setup):
        """Training loop must save checkpoint_best.pth."""
        args, model, optimizer, train_loader, val_loader, test_loader, snap_dir = tiny_setup
        experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, snap_dir)

        assert os.path.exists(snap_dir + 'checkpoint_best.pth')

    def test_creates_loss_histories(self, tiny_setup):
        """Training loop must save .train_loss and .val_loss files."""
        args, model, optimizer, train_loader, val_loader, test_loader, snap_dir = tiny_setup
        experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, snap_dir)

        assert os.path.exists(snap_dir + 'vae.train_loss')
        assert os.path.exists(snap_dir + 'vae.val_loss')
        assert os.path.exists(snap_dir + 'vae.train_re')
        assert os.path.exists(snap_dir + 'vae.train_kl')

        # Verify correct number of epochs recorded
        train_loss = torch.load(snap_dir + 'vae.train_loss', weights_only=True)
        assert len(train_loss) == 3

    def test_creates_config(self, tiny_setup):
        """Training loop must save model config."""
        args, model, optimizer, train_loader, val_loader, test_loader, snap_dir = tiny_setup
        experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, snap_dir)

        assert os.path.exists(snap_dir + 'vae.config')

    def test_creates_test_metrics(self, tiny_setup):
        """final_evaluation must save test loss metrics."""
        args, model, optimizer, train_loader, val_loader, test_loader, snap_dir = tiny_setup
        experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, snap_dir)

        assert os.path.exists(snap_dir + 'vae.test_loss')
        assert os.path.exists(snap_dir + 'vae.test_re')
        assert os.path.exists(snap_dir + 'vae.test_kl')

    def test_creates_ll_metrics_csv(self, tiny_setup):
        """final_evaluation must create ll_metrics.csv even when LL is skipped."""
        args, model, optimizer, train_loader, val_loader, test_loader, snap_dir = tiny_setup
        experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, snap_dir)

        csv_path = snap_dir + 'll_metrics.csv'
        assert os.path.exists(csv_path)

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert 'test_elbo' in df.columns
        assert 'epochs_trained' in df.columns
        assert df['epochs_trained'].iloc[0] == 3

    def test_losses_are_finite(self, tiny_setup):
        """All saved losses must be finite (no NaN or Inf)."""
        args, model, optimizer, train_loader, val_loader, test_loader, snap_dir = tiny_setup
        experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, snap_dir)

        test_loss = torch.load(snap_dir + 'vae.test_loss', weights_only=True)
        assert torch.isfinite(torch.tensor(test_loss)), f"test_loss is {test_loss}"

    def test_creates_reconstruction_images(self, tiny_setup):
        """final_evaluation must save reconstruction visualizations."""
        args, model, optimizer, train_loader, val_loader, test_loader, snap_dir = tiny_setup
        experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, snap_dir)

        assert os.path.exists(snap_dir + 'real.png')
        assert os.path.exists(snap_dir + 'reconstructions.png')


def test_model_dir_printed_to_stdout(tiny_setup, capsys):
    """Training must print MODEL_DIR: <path> for external tools to parse."""
    args, model, optimizer, train_loader, val_loader, test_loader, snap_dir = tiny_setup
    experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, snap_dir)

    captured = capsys.readouterr()
    assert 'MODEL_DIR:' in captured.out
    # The printed path must match the actual snap_dir
    for line in captured.out.splitlines():
        if line.startswith('MODEL_DIR:'):
            printed_path = line.split('MODEL_DIR:')[1].strip()
            assert printed_path == snap_dir.rstrip('/')
            break
