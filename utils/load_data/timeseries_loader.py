"""
Time series data loaders for the VAE framework.

Supported datasets:
  - ECG5000: UCR Time Series Archive, 140 timesteps, 1 feature, 5 classes
  - synthetic_timeseries: generated sine waves with trend + noise (no download)
  - csv_timeseries: legacy bring-your-own CSV — DEPRECATED in favour of tabular_timeseries
  - parquet_timeseries: legacy bring-your-own Parquet — DEPRECATED in favour of tabular_timeseries
  - tabular_timeseries: unified bring-your-own CSV/TSV/Parquet/NPY loader (univariate only)
"""

import os
import re
import sys
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


class parquet_timeseries_loader(base_load_data):
    """Loader for Parquet time-series data (calcium ingestion output).

    Expects a Parquet file where:
      - Sample columns are named t_000, t_001, ... t_{N-1} (strict 't_<digit>' pattern)
      - All other columns are metadata (subject_id, group, roi_idx,
        time_start_sec, etc.) and are dropped from the model input.

    Global MinMax normalization to [0, 1]. Deterministic 80/10/10 split.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def obtain_data(self):
        return None, None

    def load_dataset(self, **kwargs):
        import re
        import pandas as pd

        parquet_path = getattr(self.args, 'parquet_path', None)
        if not parquet_path:
            raise ValueError("--parquet_path is required for parquet_timeseries dataset")

        df = pd.read_parquet(parquet_path)
        # strict sample pattern: t_<digits>. Excludes time_start_sec, time_end_sec, etc.
        t_pat = re.compile(r"^t_\d+$")
        t_cols = sorted(c for c in df.columns if t_pat.match(c))
        if not t_cols:
            raise ValueError(f"{parquet_path}: no 't_NNN' columns found")

        x_all = df[t_cols].to_numpy(dtype=np.float32)                   # (N, T)
        N, T = x_all.shape

        x_min = float(x_all.min())
        x_max = float(x_all.max())
        x_all = (x_all - x_min) / (x_max - x_min + 1e-7)

        y_all = np.zeros(N, dtype=int)

        rng = np.random.default_rng(getattr(self.args, 'seed', 42))
        perm = rng.permutation(N)
        x_all = x_all[perm]
        y_all = y_all[perm]

        n_train = int(0.8 * N)
        n_val   = int(0.1 * N)
        x_train, y_train = x_all[:n_train],               y_all[:n_train]
        x_val,   y_val   = x_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
        x_test,  y_test  = x_all[n_train + n_val:],        y_all[n_train + n_val:]

        feat_dim = 1
        self.args.seq_len             = T
        self.args.feat_dim            = feat_dim
        self.args.input_size          = [feat_dim, T]
        self.args.input_type          = 'continuous'
        self.args.use_logit           = False
        self.args.dynamic_binarization = False
        self.args.training_set_size   = n_train

        x_train = x_train.reshape(-1, feat_dim * T)
        x_val   = x_val.reshape(-1,   feat_dim * T)
        x_test  = x_test.reshape(-1,  feat_dim * T)

        print("Parquet TimeSeries data stats:")
        print(f"  Source: {parquet_path}")
        print(f"  Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        print(f"  Seq length: {T}, Features: {feat_dim}")
        print(f"  Value range before normalisation: [{x_min:.4f}, {x_max:.4f}]")

        train_loader, val_loader, test_loader = self.post_processing(
            x_train, x_val, x_test, y_train, y_val, y_test, **kwargs)

        return train_loader, val_loader, test_loader, self.args


# ======================================================================================================================
# Unified tabular time-series loader (CSV / TSV / Parquet / NPY)
# ======================================================================================================================
class tabular_timeseries_loader(base_load_data):
    """Unified bring-your-own time-series loader for CSV/TSV/Parquet/NPY input.

    Replaces the project-specific csv_timeseries_loader and
    parquet_timeseries_loader. Driven entirely by --ts_* CLI flags;
    no hardcoded column conventions. Univariate only (feat_dim=1);
    multivariate is out of scope for this loader.

    See API.md ("Tabular time-series — bring your own data") for the
    full flag and input-contract reference.
    """

    _EXT_TO_FORMAT = {
        '.csv': 'csv', '.tsv': 'csv',
        '.parquet': 'parquet', '.pq': 'parquet',
        '.npy': 'npy',
    }

    def __init__(self, args, **kwargs):
        super().__init__(args)

    def obtain_data(self):
        # tabular_timeseries overrides load_dataset() entirely; obtain_data
        # exists only to satisfy the abstractmethod from base_load_data.
        return None, None

    # ----------------------------------------------------------------
    # Format / separator resolution
    # ----------------------------------------------------------------
    def _resolve_format(self, path):
        """Return the effective file format ('csv'/'parquet'/'npy').

        Honour `args.ts_format` if set to a concrete value; otherwise
        infer from extension. Exit with a clear error if the extension
        is unknown and no override was given.
        """
        explicit = getattr(self.args, 'ts_format', 'auto')
        if explicit and explicit != 'auto':
            return explicit
        ext = os.path.splitext(str(path))[1].lower()
        fmt = self._EXT_TO_FORMAT.get(ext)
        if fmt is None:
            sys.exit(
                f"--ts_path '{path}' has unrecognised extension '{ext}'. "
                f"Pass --ts_format {{csv,parquet,npy}} or rename the file. "
                f"Supported extensions: {sorted(self._EXT_TO_FORMAT.keys())}"
            )
        return fmt

    def _effective_csv_sep(self, path):
        """Default --ts_csv_sep to '\\t' for .tsv files when user kept the default ','.

        Lets users name a file `data.tsv` and not have to remember
        `--ts_csv_sep "\\t"`. Any non-default user value wins.
        """
        sep = getattr(self.args, 'ts_csv_sep', ',')
        if sep == ',' and str(path).lower().endswith('.tsv'):
            return '\t'
        return sep

    # ----------------------------------------------------------------
    # File loading
    # ----------------------------------------------------------------
    def _load_dataframe(self, path, fmt):
        """Read a csv/tsv or parquet file into a DataFrame.

        npy files do not go through this method — they're loaded
        directly in load_dataset() via np.load.
        """
        import pandas as pd
        if fmt == 'csv':
            return pd.read_csv(path, sep=self._effective_csv_sep(path))
        if fmt == 'parquet':
            return pd.read_parquet(path)
        sys.exit(f"_load_dataframe: unsupported format '{fmt}' (expected 'csv' or 'parquet')")

    # ----------------------------------------------------------------
    # Column selection — regex or explicit list
    # ----------------------------------------------------------------
    def _select_value_cols(self, df):
        """Pick the columns named as time-sample values per --ts_value_cols.

        Returns (x: np.ndarray of shape (N, T), cols: list[str] in sorted order).
        Sort is by ASCII column name — deterministic across pandas versions.

        Pattern semantics:
        - If the flag value contains a comma, treat as explicit list.
        - Otherwise, treat as regex pattern compiled with re.match.
        """
        pattern = getattr(self.args, 'ts_value_cols', None)
        if pattern is None:
            preview = list(df.columns[:5])
            sys.exit(
                "--ts_value_cols is required for csv/parquet inputs (regex pattern "
                f"or comma-separated list). First columns in file: {preview}"
            )

        if ',' in pattern:
            requested = [c.strip() for c in pattern.split(',') if c.strip()]
            missing = [c for c in requested if c not in df.columns]
            if missing:
                sys.exit(
                    f"--ts_value_cols: column(s) {missing} not in file. "
                    f"Available: {list(df.columns)}"
                )
            cols = sorted(requested)
        else:
            try:
                rx = re.compile(pattern)
            except re.error as exc:
                sys.exit(f"--ts_value_cols: invalid regex '{pattern}' ({exc})")
            cols = sorted(c for c in df.columns if rx.match(str(c)))
            if not cols:
                sys.exit(
                    f"--ts_value_cols regex '{pattern}' matched 0 columns. "
                    f"Columns in file: {list(df.columns)}"
                )

        x = df[cols].to_numpy(dtype=np.float32)
        return x, cols

    # ----------------------------------------------------------------
    # NPY direct loading
    # ----------------------------------------------------------------
    def _load_npy(self, path):
        """Load a .npy file as a 2D float32 array (N, T).

        --ts_value_cols and --ts_label_col are silently ignored for npy
        input; npy has no column names and no built-in label channel.
        Users who need labels should use csv/parquet.
        """
        arr = np.load(path)
        if arr.ndim != 2:
            sys.exit(
                f"--ts_path '{path}' loaded a {arr.ndim}D npy array, expected 2D (N, T). "
                f"shape: {arr.shape}"
            )
        return arr.astype(np.float32, copy=False)

    # ----------------------------------------------------------------
    # Label extraction
    # ----------------------------------------------------------------
    def _extract_labels(self, df, n):
        """Return an int64 label array of length n.

        - --ts_label_col not set → all zeros (anomaly-detection / unsupervised default).
        - --ts_label_col present in df → pd.factorize maps string/object to ints.
        """
        import pandas as pd
        col = getattr(self.args, 'ts_label_col', None)
        if col is None:
            return np.zeros(n, dtype=np.int64)
        if col not in df.columns:
            sys.exit(
                f"--ts_label_col '{col}' not found in file. "
                f"Available columns: {list(df.columns)}"
            )
        codes, _ = pd.factorize(df[col])
        return codes.astype(np.int64)

    # ----------------------------------------------------------------
    # Path resolution + splits
    # ----------------------------------------------------------------
    def _resolve_paths(self):
        """Decide between single-file and pre-split modes.

        Returns ('single', single_path) or ('trio', [train, val, test]).
        Exits with a clear error on collision, partial trio, or neither.
        """
        single = getattr(self.args, 'ts_path', None)
        trio = [getattr(self.args, f'ts_{s}_path', None) for s in ('train', 'val', 'test')]
        any_trio = any(p is not None for p in trio)
        all_trio = all(p is not None for p in trio)

        if single and any_trio:
            sys.exit(
                "specify either --ts_path (single file, auto-split) OR "
                "all three of --ts_train_path/--ts_val_path/--ts_test_path "
                "(pre-split), not both"
            )
        if any_trio and not all_trio:
            missing = [s for s, p in zip(('train', 'val', 'test'), trio) if p is None]
            sys.exit(
                f"pre-split mode requires --ts_train_path AND --ts_val_path AND "
                f"--ts_test_path; missing: {missing}"
            )
        if not single and not all_trio:
            sys.exit(
                "no input path: specify --ts_path (single file) or "
                "--ts_train_path / --ts_val_path / --ts_test_path (pre-split)"
            )

        if single:
            return ('single', single)
        return ('trio', trio)

    def _parse_split(self, spec):
        """Parse '0.8/0.1/0.1' → [0.8, 0.1, 0.1] with sum-to-1 validation."""
        parts = spec.split('/')
        if len(parts) != 3:
            sys.exit(
                f"--ts_split must have 3 ratios separated by '/', got {len(parts)}: '{spec}'"
            )
        try:
            ratios = [float(p) for p in parts]
        except ValueError:
            sys.exit(f"--ts_split contains non-numeric ratio: '{spec}'")
        if abs(sum(ratios) - 1.0) > 1e-6:
            sys.exit(
                f"--ts_split ratios must sum to 1.0 (±1e-6), got {sum(ratios):.6f}: '{spec}'"
            )
        return ratios

    def _split_single(self, x, y, spec):
        """Deterministic shuffle + ratio split of a single (x, y) into 3 splits.

        Uses np.random.default_rng(self.args.seed) so two runs with the
        same seed produce identical splits regardless of global RNG state.
        """
        ratios = self._parse_split(spec)
        n = len(x)
        rng = np.random.default_rng(getattr(self.args, 'seed', 42))
        perm = rng.permutation(n)
        x = x[perm]
        y = y[perm]

        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        x_train, y_train = x[:n_train], y[:n_train]
        x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
        x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    # ----------------------------------------------------------------
    # Normalisation
    # ----------------------------------------------------------------
    def _normalise(self, x_train, x_val, x_test):
        """Apply the chosen --ts_normalise mode.

        Statistics are computed from x_train only and applied to all three
        splits — prevents test-set leakage in modes that need a global
        scale (global_minmax, zscore). per_sample_minmax is row-local so
        leakage is moot. 'none' passes through unchanged.
        """
        mode = getattr(self.args, 'ts_normalise', 'global_minmax')
        if mode == 'global_minmax':
            lo = float(x_train.min())
            hi = float(x_train.max())
            denom = hi - lo + 1e-7
            return tuple((x - lo) / denom for x in (x_train, x_val, x_test))
        if mode == 'per_sample_minmax':
            def f(x):
                lo = x.min(axis=1, keepdims=True)
                hi = x.max(axis=1, keepdims=True)
                return (x - lo) / (hi - lo + 1e-7)
            return tuple(f(x) for x in (x_train, x_val, x_test))
        if mode == 'zscore':
            mu = float(x_train.mean())
            sigma = float(x_train.std())
            return tuple((x - mu) / (sigma + 1e-7) for x in (x_train, x_val, x_test))
        if mode == 'none':
            return x_train, x_val, x_test
        sys.exit(
            f"--ts_normalise must be one of "
            f"{{global_minmax, per_sample_minmax, zscore, none}}, got '{mode}'"
        )
