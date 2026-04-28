"""Tests for utils/load_data/timeseries_loader.tabular_timeseries_loader.

The unified tabular_timeseries loader replaces the per-format
csv_timeseries / parquet_timeseries loaders. It accepts csv/tsv/parquet/npy
input via a single set of --ts_* flags, with regex or explicit-list column
selection, optional label column, configurable splits, and four
normalisation modes.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_args(**overrides):
    """Minimal Namespace mirroring run.py argparse for tabular_timeseries flags."""
    base = dict(
        # --ts_* group
        ts_path=None,
        ts_train_path=None,
        ts_val_path=None,
        ts_test_path=None,
        ts_format='auto',
        ts_csv_sep=',',
        ts_value_cols=None,
        ts_label_col=None,
        ts_split='0.8/0.1/0.1',
        ts_normalise='global_minmax',
        # standard fields touched by base_load_data + post_processing
        seed=42,
        batch_size=16,
        test_batch_size=16,
        training_set_size=0,
        use_logit=False,
        continuous=True,
        lambd=1e-4,
        dynamic_binarization=False,
        dataset_name='tabular_timeseries',
        prior='standard',
        number_components=10,
        pseudoinputs_mean=-0.05,
        pseudoinputs_std=0.01,
        use_training_data_init=False,
        # populated by loader; pre-fill for completeness
        seq_len=None,
        feat_dim=None,
        input_size=None,
        input_type=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ======================================================================
# Format auto-detection
# ======================================================================
class TestFormatDetection:
    def test_auto_detect_csv(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='data.csv')
        loader = tabular_timeseries_loader(args)
        assert loader._resolve_format('data.csv') == 'csv'

    def test_auto_detect_tsv(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='data.tsv')
        loader = tabular_timeseries_loader(args)
        assert loader._resolve_format('data.tsv') == 'csv'

    def test_auto_detect_parquet(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='data.parquet')
        loader = tabular_timeseries_loader(args)
        assert loader._resolve_format('data.parquet') == 'parquet'

    def test_auto_detect_npy(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='data.npy')
        loader = tabular_timeseries_loader(args)
        assert loader._resolve_format('data.npy') == 'npy'

    def test_explicit_format_override(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='data.weird', ts_format='parquet')
        loader = tabular_timeseries_loader(args)
        # Override wins regardless of extension.
        assert loader._resolve_format('data.weird') == 'parquet'

    def test_unknown_extension_no_override(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='data.weird', ts_format='auto')
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader._resolve_format('data.weird')

    def test_tsv_extension_implies_tab_separator_when_default_sep(self):
        """A .tsv file with default --ts_csv_sep ',' should fall back to '\\t'.

        Reasoning: TSV's whole point is tab separation; if user named the file
        .tsv but left the default separator, treat as TSV automatically.
        """
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='data.tsv', ts_csv_sep=',')
        loader = tabular_timeseries_loader(args)
        # _effective_csv_sep returns '\t' when path ends in .tsv and user kept default ','
        assert loader._effective_csv_sep('data.tsv') == '\t'

    def test_csv_extension_keeps_user_sep(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='data.csv', ts_csv_sep=';')
        loader = tabular_timeseries_loader(args)
        assert loader._effective_csv_sep('data.csv') == ';'
