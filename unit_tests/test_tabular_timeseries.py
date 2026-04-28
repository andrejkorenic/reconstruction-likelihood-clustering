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


# ======================================================================
# Column selection (csv / parquet)
# ======================================================================
def _csv_with_value_and_meta(path, n_rows=8, n_t=5, sep=','):
    """Build a small TSV/CSV fixture: t_0..t_{n_t-1} + extra metadata cols."""
    rng = np.random.default_rng(0)
    cols = {f't_{i}': rng.random(n_rows).astype(np.float32) for i in range(n_t)}
    cols['subject_id'] = [f's{i % 3}' for i in range(n_rows)]
    cols['group'] = ['ALS' if i % 2 == 0 else 'CTRL' for i in range(n_rows)]
    cols['time_start_sec'] = rng.random(n_rows)  # collision trap for naive 't_' prefix
    df = pd.DataFrame(cols)
    df.to_csv(path, sep=sep, index=False)
    return df, [f't_{i}' for i in range(n_t)]


def _parquet_with_value_and_meta(path, n_rows=8, n_t=5):
    rng = np.random.default_rng(1)
    cols = {f't_{i}': rng.random(n_rows).astype(np.float32) for i in range(n_t)}
    cols['subject_id'] = [f's{i % 3}' for i in range(n_rows)]
    cols['group'] = ['ALS' if i % 2 == 0 else 'CTRL' for i in range(n_rows)]
    cols['time_start_sec'] = rng.random(n_rows)
    df = pd.DataFrame(cols)
    df.to_parquet(path)
    return df, [f't_{i}' for i in range(n_t)]


class TestColumnSelection:
    def test_regex_value_cols_csv(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        df, expected = _csv_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$')
        loader = tabular_timeseries_loader(args)
        df_loaded = loader._load_dataframe(str(path), 'csv')
        x, cols = loader._select_value_cols(df_loaded)
        assert cols == expected
        assert x.shape == (8, 5)
        assert x.dtype == np.float32

    def test_regex_value_cols_parquet(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.parquet"
        df, expected = _parquet_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$')
        loader = tabular_timeseries_loader(args)
        df_loaded = loader._load_dataframe(str(path), 'parquet')
        x, cols = loader._select_value_cols(df_loaded)
        assert cols == expected
        assert x.shape == (8, 5)

    def test_explicit_list_value_cols(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        df, _ = _csv_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols='t_2,t_0,t_4')
        loader = tabular_timeseries_loader(args)
        df_loaded = loader._load_dataframe(str(path), 'csv')
        x, cols = loader._select_value_cols(df_loaded)
        # Sorted by name → deterministic
        assert cols == ['t_0', 't_2', 't_4']
        assert x.shape == (8, 3)

    def test_regex_no_match(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _csv_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^xyz_\d+$')
        loader = tabular_timeseries_loader(args)
        df = loader._load_dataframe(str(path), 'csv')
        with pytest.raises(SystemExit):
            loader._select_value_cols(df)

    def test_explicit_list_missing_col(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _csv_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols='t_0,nonexistent_col,t_2')
        loader = tabular_timeseries_loader(args)
        df = loader._load_dataframe(str(path), 'csv')
        with pytest.raises(SystemExit):
            loader._select_value_cols(df)

    def test_missing_value_cols_for_csv(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _csv_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols=None)
        loader = tabular_timeseries_loader(args)
        df = loader._load_dataframe(str(path), 'csv')
        with pytest.raises(SystemExit):
            loader._select_value_cols(df)

    def test_value_cols_deterministic_order(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        # write columns in scrambled order
        path = tmp_path / "d.csv"
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            't_3': rng.random(4),
            't_0': rng.random(4),
            't_2': rng.random(4),
            't_1': rng.random(4),
        })
        df.to_csv(path, index=False)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$')
        loader = tabular_timeseries_loader(args)
        df_loaded = loader._load_dataframe(str(path), 'csv')
        _, cols = loader._select_value_cols(df_loaded)
        assert cols == ['t_0', 't_1', 't_2', 't_3']

    def test_csv_sep_tsv(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.tsv"
        df, expected = _csv_with_value_and_meta(path, sep='\t')
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$')
        loader = tabular_timeseries_loader(args)
        df_loaded = loader._load_dataframe(str(path), 'csv')
        # If sep is wrong, the entire row collapses into a single column
        # and our regex won't match anything.
        x, cols = loader._select_value_cols(df_loaded)
        assert cols == expected
        assert x.shape == (8, 5)


# ======================================================================
# NPY input
# ======================================================================
class TestNpyInput:
    def test_npy_2d_shape(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        arr = np.random.default_rng(0).random((40, 20)).astype(np.float32)
        p = tmp_path / "d.npy"
        np.save(p, arr)
        args = _make_args(ts_path=str(p))
        loader = tabular_timeseries_loader(args)
        x = loader._load_npy(str(p))
        assert x.shape == (40, 20)
        assert x.dtype == np.float32

    def test_npy_1d_fails(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        arr = np.array([1.0, 2.0, 3.0])
        p = tmp_path / "d.npy"
        np.save(p, arr)
        args = _make_args(ts_path=str(p))
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader._load_npy(str(p))

    def test_npy_3d_fails(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        arr = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
        p = tmp_path / "d.npy"
        np.save(p, arr)
        args = _make_args(ts_path=str(p))
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader._load_npy(str(p))

    def test_npy_dtype_coerced_to_float32(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        arr = np.random.default_rng(0).random((5, 7)).astype(np.float64)
        p = tmp_path / "d.npy"
        np.save(p, arr)
        args = _make_args(ts_path=str(p))
        loader = tabular_timeseries_loader(args)
        x = loader._load_npy(str(p))
        assert x.dtype == np.float32


# ======================================================================
# Optional label column
# ======================================================================
class TestLabelColumn:
    def test_no_label_col_default_zeros(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _csv_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$', ts_label_col=None)
        loader = tabular_timeseries_loader(args)
        df = loader._load_dataframe(str(path), 'csv')
        y = loader._extract_labels(df, n=len(df))
        assert y.dtype == np.int64
        assert (y == 0).all()
        assert len(y) == len(df)

    def test_label_col_strings(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        df_orig, _ = _csv_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$', ts_label_col='group')
        loader = tabular_timeseries_loader(args)
        df = loader._load_dataframe(str(path), 'csv')
        y = loader._extract_labels(df, n=len(df))
        # 'group' alternates ALS/CTRL; pd.factorize gives ALS=0, CTRL=1 (first-seen order)
        assert y.dtype == np.int64
        assert set(y.tolist()) == {0, 1}
        # Same string should always factorise to the same int
        for i, label in enumerate(df_orig['group']):
            for j, other in enumerate(df_orig['group']):
                if label == other:
                    assert y[i] == y[j]

    def test_label_col_ints(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            't_0': rng.random(6).astype(np.float32),
            't_1': rng.random(6).astype(np.float32),
            'cls': [3, 1, 4, 1, 5, 9],
        })
        df.to_csv(path, index=False)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$', ts_label_col='cls')
        loader = tabular_timeseries_loader(args)
        df_loaded = loader._load_dataframe(str(path), 'csv')
        y = loader._extract_labels(df_loaded, n=len(df_loaded))
        assert y.dtype == np.int64
        # factorize re-codes by first-seen order: 3→0, 1→1, 4→2, 5→3, 9→4
        assert y.tolist() == [0, 1, 2, 1, 3, 4]

    def test_label_col_missing(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _csv_with_value_and_meta(path)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$',
                          ts_label_col='nonexistent')
        loader = tabular_timeseries_loader(args)
        df = loader._load_dataframe(str(path), 'csv')
        with pytest.raises(SystemExit):
            loader._extract_labels(df, n=len(df))


# ======================================================================
# Splits — single-file ratio + pre-split 3-path
# ======================================================================
class TestSplits:
    def test_collision_ts_path_and_train_path(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path='a.csv', ts_train_path='b.csv',
                          ts_val_path='c.csv', ts_test_path='d.csv')
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader._resolve_paths()

    def test_pre_split_partial_only_two_paths(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_train_path='b.csv', ts_val_path='c.csv',
                          ts_test_path=None)
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader._resolve_paths()

    def test_pre_split_all_three_paths(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_train_path='b.csv', ts_val_path='c.csv',
                          ts_test_path='d.csv')
        loader = tabular_timeseries_loader(args)
        kind, paths = loader._resolve_paths()
        assert kind == 'trio'
        assert paths == ['b.csv', 'c.csv', 'd.csv']

    def test_neither_ts_path_nor_trio(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_path=None)
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader._resolve_paths()

    def test_split_ratio_default(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_split='0.8/0.1/0.1')
        loader = tabular_timeseries_loader(args)
        ratios = loader._parse_split('0.8/0.1/0.1')
        assert ratios == pytest.approx([0.8, 0.1, 0.1])

    def test_split_ratio_custom(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_split='0.7/0.15/0.15')
        loader = tabular_timeseries_loader(args)
        ratios = loader._parse_split('0.7/0.15/0.15')
        assert ratios == pytest.approx([0.7, 0.15, 0.15])

    def test_split_ratio_must_sum_to_one(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_split='0.5/0.3/0.3')  # sums to 1.1
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader._parse_split('0.5/0.3/0.3')

    def test_split_ratio_wrong_count(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_split='0.8/0.2')
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader._parse_split('0.8/0.2')

    def test_single_path_default_split_sizes(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        rng = np.random.default_rng(0)
        x = rng.random((100, 10)).astype(np.float32)
        y = np.zeros(100, dtype=np.int64)
        args = _make_args(ts_split='0.8/0.1/0.1', seed=42)
        loader = tabular_timeseries_loader(args)
        (xt, yt), (xv, yv), (xe, ye) = loader._split_single(x, y, '0.8/0.1/0.1')
        assert len(xt) == 80
        assert len(xv) == 10
        assert len(xe) == 10
        assert len(xt) + len(xv) + len(xe) == 100

    def test_single_path_custom_split(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        rng = np.random.default_rng(0)
        x = rng.random((100, 10)).astype(np.float32)
        y = np.zeros(100, dtype=np.int64)
        args = _make_args(ts_split='0.7/0.15/0.15', seed=42)
        loader = tabular_timeseries_loader(args)
        (xt, _), (xv, _), (xe, _) = loader._split_single(x, y, '0.7/0.15/0.15')
        assert len(xt) == 70
        assert len(xv) == 15
        assert len(xe) == 15

    def test_split_determinism_seed(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        rng = np.random.default_rng(0)
        x = rng.random((100, 10)).astype(np.float32)
        y = np.arange(100, dtype=np.int64)
        args = _make_args(seed=42)
        loader1 = tabular_timeseries_loader(args)
        loader2 = tabular_timeseries_loader(args)
        (xt1, yt1), _, _ = loader1._split_single(x, y, '0.8/0.1/0.1')
        (xt2, yt2), _, _ = loader2._split_single(x, y, '0.8/0.1/0.1')
        np.testing.assert_array_equal(xt1, xt2)
        np.testing.assert_array_equal(yt1, yt2)

    def test_split_different_seed(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        rng = np.random.default_rng(0)
        x = rng.random((100, 10)).astype(np.float32)
        y = np.arange(100, dtype=np.int64)
        args1 = _make_args(seed=1)
        args2 = _make_args(seed=2)
        loader1 = tabular_timeseries_loader(args1)
        loader2 = tabular_timeseries_loader(args2)
        (_, yt1), _, _ = loader1._split_single(x, y, '0.8/0.1/0.1')
        (_, yt2), _, _ = loader2._split_single(x, y, '0.8/0.1/0.1')
        # With 100 samples and different seeds, the two permutations should differ
        # almost surely. Test by checking the train-set indices are not identical.
        assert not np.array_equal(yt1, yt2)
