"""
Time series data loaders for the VAE framework.

Supported datasets:
  - ECG5000: UCR Time Series Archive, 140 timesteps, 1 feature, 5 classes
  - synthetic_timeseries: generated sine waves with trend + noise (no download)
"""

import os
import zipfile
import numpy as np
import torch
import torch.utils.data as data_utils
import urllib.request

from .base_load_data import base_load_data


class ecg5000_loader(base_load_data):
    """ECG5000 from the UCR Time Series Classification Archive.

    5000 heartbeat samples, 140 timesteps, 1 feature, 5 classes.
    Standard UCR split: 500 train, 4500 test.
    We re-split: 4000 train, 500 val, 500 test.
    MinMax normalized to [0, 1].
    """
    URL = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def obtain_data(self):
        return None, None  # not used — load_dataset is overridden

    def _download_if_needed(self):
        data_dir = os.path.join('datasets', 'ECG5000')
        # UCR archive provides .txt (space-separated) files
        train_path = os.path.join(data_dir, 'ECG5000_TRAIN.txt')
        test_path = os.path.join(data_dir, 'ECG5000_TEST.txt')

        if os.path.exists(train_path) and os.path.exists(test_path):
            return train_path, test_path

        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, 'ECG5000.zip')
        print(f"Downloading ECG5000 from {self.URL}...")
        urllib.request.urlretrieve(self.URL, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)

        # the zip may extract into a subdirectory — move files up
        for root, dirs, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.txt') and root != data_dir:
                    os.rename(os.path.join(root, f), os.path.join(data_dir, f))

        os.remove(zip_path)
        return train_path, test_path

    def _parse_ucr_txt(self, path):
        """Parse UCR text format: each row is label, val1, val2, ..., valN (space-separated)."""
        raw = np.loadtxt(path)
        labels = raw[:, 0].astype(int)
        data = raw[:, 1:]
        return data, labels

    def load_dataset(self, **kwargs):
        train_path, test_path = self._download_if_needed()

        x_train_raw, y_train_raw = self._parse_ucr_txt(train_path)
        x_test_raw, y_test_raw = self._parse_ucr_txt(test_path)

        # combine and re-split: 4000 train, 500 val, 500 test
        x_all = np.concatenate([x_train_raw, x_test_raw], axis=0)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)

        np.random.seed(42)
        perm = np.random.permutation(len(x_all))
        x_all, y_all = x_all[perm], y_all[perm]

        x_train = x_all[:4000]
        y_train = y_all[:4000]
        x_val = x_all[4000:4500]
        y_val = y_all[4000:4500]
        x_test = x_all[4500:]
        y_test = y_all[4500:]

        # MinMax normalization to [0, 1] (fit on train only)
        self._min = x_train.min()
        self._max = x_train.max()
        x_train = (x_train - self._min) / (self._max - self._min + 1e-7)
        x_val = (x_val - self._min) / (self._max - self._min + 1e-7)
        x_test = (x_test - self._min) / (self._max - self._min + 1e-7)

        # runtime fields
        seq_len = x_train.shape[1]
        feat_dim = 1
        self.args.seq_len = seq_len
        self.args.feat_dim = feat_dim
        self.args.input_size = [feat_dim, seq_len]
        self.args.input_type = 'continuous'
        self.args.use_logit = False
        self.args.dynamic_binarization = False
        self.args.training_set_size = len(x_train)

        # flatten: (N, seq_len) → (N, feat_dim * seq_len) = (N, 140)
        x_train = x_train.reshape(-1, feat_dim * seq_len)
        x_val = x_val.reshape(-1, feat_dim * seq_len)
        x_test = x_test.reshape(-1, feat_dim * seq_len)

        # remap labels to 0-based
        unique = np.unique(y_train)
        label_map = {old: new for new, old in enumerate(unique)}
        y_train = np.array([label_map[y] for y in y_train])
        y_val = np.array([label_map[y] for y in y_val])
        y_test = np.array([label_map[y] for y in y_test])

        print("ECG5000 data stats:")
        print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        print(f"  Seq length: {seq_len}, Features: {feat_dim}, Classes: {len(unique)}")

        train_loader, val_loader, test_loader = self.post_processing(
            x_train, x_val, x_test, y_train, y_val, y_test, **kwargs)

        return train_loader, val_loader, test_loader, self.args


class synthetic_timeseries_loader(base_load_data):
    """Synthetic time series: sine waves with varying frequency, trend, and noise.

    Generated in-code, no download needed. Good for quick architecture verification.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def obtain_data(self):
        return None, None

    def load_dataset(self, **kwargs):
        seq_len = getattr(self.args, 'seq_len', 50)
        feat_dim = getattr(self.args, 'feat_dim', 1)
        n_samples = 5000
        n_classes = 5

        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, seq_len)

        x_all, y_all = [], []
        for cls in range(n_classes):
            n = n_samples // n_classes
            freq = 1.0 + cls * 0.5                              # different frequency per class
            phase = np.random.uniform(0, 2 * np.pi, (n, 1))
            trend = np.random.uniform(-0.1, 0.1, (n, 1)) * t
            noise = np.random.normal(0, 0.05, (n, seq_len))
            signal = np.sin(freq * t + phase) + trend + noise   # (n, seq_len)
            x_all.append(signal)
            y_all.append(np.full(n, cls))

        x_all = np.concatenate(x_all, axis=0).astype(np.float32)
        y_all = np.concatenate(y_all, axis=0).astype(int)

        # shuffle
        perm = np.random.permutation(len(x_all))
        x_all, y_all = x_all[perm], y_all[perm]

        # MinMax to [0, 1]
        x_min, x_max = x_all.min(), x_all.max()
        x_all = (x_all - x_min) / (x_max - x_min + 1e-7)

        # split
        x_train = x_all[:4000]
        y_train = y_all[:4000]
        x_val = x_all[4000:4500]
        y_val = y_all[4000:4500]
        x_test = x_all[4500:]
        y_test = y_all[4500:]

        # runtime fields
        self.args.seq_len = seq_len
        self.args.feat_dim = feat_dim
        self.args.input_size = [feat_dim, seq_len]
        self.args.input_type = 'continuous'
        self.args.use_logit = False
        self.args.dynamic_binarization = False
        self.args.training_set_size = len(x_train)

        # flatten for pipeline
        x_train = x_train.reshape(-1, feat_dim * seq_len)
        x_val = x_val.reshape(-1, feat_dim * seq_len)
        x_test = x_test.reshape(-1, feat_dim * seq_len)

        print("Synthetic timeseries data stats:")
        print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        print(f"  Seq length: {seq_len}, Features: {feat_dim}, Classes: {n_classes}")

        train_loader, val_loader, test_loader = self.post_processing(
            x_train, x_val, x_test, y_train, y_val, y_test, **kwargs)

        return train_loader, val_loader, test_loader, self.args


class csv_timeseries_loader(base_load_data):
    """Loader for generic TSV calcium imaging data.

    Expects a TSV file with:
      - 3 leading metadata columns (any names) — discarded
      - Remaining columns: float timestep values (one row = one time series)

    No download needed; file path must be in args.csv_path.
    Global MinMax normalization to [0, 1] across the full dataset.
    Deterministic 80 / 10 / 10 train / val / test split (seeded by args.seed).
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def obtain_data(self):
        return None, None  # not used — load_dataset is fully overridden

    def load_dataset(self, **kwargs):
        import pandas as pd

        csv_path = getattr(self.args, 'csv_path', None)
        if not csv_path:
            raise ValueError("--csv_path is required for the csv_timeseries dataset")

        df = pd.read_csv(csv_path, sep='\t')

        # drop leading metadata columns (configurable via --csv_meta_cols)
        n_meta = getattr(self.args, 'csv_meta_cols', 2)
        x_all = df.iloc[:, n_meta:].values.astype(np.float32)   # (N, T)
        N, T = x_all.shape

        # global MinMax normalization → [0, 1]
        x_min = float(x_all.min())
        x_max = float(x_all.max())
        x_all = (x_all - x_min) / (x_max - x_min + 1e-7)

        # dummy labels — no class information in this dataset
        y_all = np.zeros(N, dtype=int)

        # deterministic shuffle + split
        rng = np.random.default_rng(getattr(self.args, 'seed', 42))
        perm = rng.permutation(N)
        x_all = x_all[perm]
        y_all = y_all[perm]

        n_train = int(0.8 * N)
        n_val   = int(0.1 * N)
        x_train, y_train = x_all[:n_train],               y_all[:n_train]
        x_val,   y_val   = x_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
        x_test,  y_test  = x_all[n_train + n_val:],        y_all[n_train + n_val:]

        # set runtime fields consumed by the rest of the pipeline
        feat_dim = 1
        self.args.seq_len             = T
        self.args.feat_dim            = feat_dim
        self.args.input_size          = [feat_dim, T]
        self.args.input_type          = 'continuous'
        self.args.use_logit           = False
        self.args.dynamic_binarization = False
        self.args.training_set_size   = n_train

        # flatten to 1-D vectors: (N, T) → (N, feat_dim * T)  (feat_dim=1, so no-op here)
        x_train = x_train.reshape(-1, feat_dim * T)
        x_val   = x_val.reshape(-1,   feat_dim * T)
        x_test  = x_test.reshape(-1,  feat_dim * T)

        print("CSV TimeSeries data stats:")
        print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        print(f"  Seq length: {T}, Features: {feat_dim}")
        print(f"  Value range before normalisation: [{x_min:.4f}, {x_max:.4f}]")

        train_loader, val_loader, test_loader = self.post_processing(
            x_train, x_val, x_test, y_train, y_val, y_test, **kwargs)

        return train_loader, val_loader, test_loader, self.args
