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


# ======================================================================
# Normalisation
# ======================================================================
class TestNormalisation:
    def _three_splits(self):
        rng = np.random.default_rng(0)
        x_train = rng.random((20, 5)).astype(np.float32) * 2.0  # range [0, 2)
        x_val = rng.random((5, 5)).astype(np.float32) * 3.0     # exceeds train range
        x_test = rng.random((5, 5)).astype(np.float32) * 0.5    # below train range
        return x_train, x_val, x_test

    def test_global_minmax_train_in_unit_interval(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_normalise='global_minmax')
        loader = tabular_timeseries_loader(args)
        x_train, x_val, x_test = self._three_splits()
        xt, xv, xe = loader._normalise(x_train.copy(), x_val.copy(), x_test.copy())
        assert xt.min() >= 0.0 and xt.max() <= 1.0 + 1e-6

    def test_global_minmax_does_not_use_test_data(self):
        """Test data outside the train min/max should map outside [0, 1]."""
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_normalise='global_minmax')
        loader = tabular_timeseries_loader(args)
        x_train = np.array([[0.0, 1.0], [0.5, 1.0]], dtype=np.float32)
        x_val = np.array([[2.0, -1.0]], dtype=np.float32)
        x_test = np.array([[3.0, -2.0]], dtype=np.float32)
        xt, xv, xe = loader._normalise(x_train.copy(), x_val.copy(), x_test.copy())
        # train min=0, max=1 → val 2 maps to 2.0, val -1 maps to -1.0
        assert xv[0, 0] == pytest.approx(2.0, rel=1e-5)
        assert xv[0, 1] < 0.0
        assert xe[0, 0] > 1.0

    def test_per_sample_minmax(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_normalise='per_sample_minmax')
        loader = tabular_timeseries_loader(args)
        x_train, x_val, x_test = self._three_splits()
        xt, xv, xe = loader._normalise(x_train.copy(), x_val.copy(), x_test.copy())
        for x in (xt, xv, xe):
            for row in x:
                assert row.min() == pytest.approx(0.0, abs=1e-5)
                assert row.max() == pytest.approx(1.0, abs=1e-5)

    def test_zscore(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_normalise='zscore')
        loader = tabular_timeseries_loader(args)
        x_train, x_val, x_test = self._three_splits()
        xt, _, _ = loader._normalise(x_train.copy(), x_val.copy(), x_test.copy())
        assert xt.mean() == pytest.approx(0.0, abs=1e-5)
        assert xt.std() == pytest.approx(1.0, abs=1e-3)

    def test_none_passthrough(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_normalise='none')
        loader = tabular_timeseries_loader(args)
        x_train, x_val, x_test = self._three_splits()
        xt, xv, xe = loader._normalise(x_train.copy(), x_val.copy(), x_test.copy())
        np.testing.assert_array_equal(xt, x_train)
        np.testing.assert_array_equal(xv, x_val)
        np.testing.assert_array_equal(xe, x_test)

    def test_unknown_normalise_mode(self):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        args = _make_args(ts_normalise='banana')
        loader = tabular_timeseries_loader(args)
        x_train, x_val, x_test = self._three_splits()
        with pytest.raises(SystemExit):
            loader._normalise(x_train, x_val, x_test)


# ======================================================================
# Full load_dataset orchestration
# ======================================================================
def _build_csv_dataset(path, n_rows=100, n_t=10):
    rng = np.random.default_rng(0)
    cols = {f't_{i}': rng.random(n_rows).astype(np.float32) for i in range(n_t)}
    cols['group'] = ['A' if i % 3 == 0 else 'B' for i in range(n_rows)]
    pd.DataFrame(cols).to_csv(path, index=False)


class TestLoadDatasetOrchestration:
    def test_single_file_full_load(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _build_csv_dataset(path, n_rows=100, n_t=10)
        args = _make_args(
            ts_path=str(path),
            ts_value_cols=r'^t_\d+$',
            ts_normalise='global_minmax',
            seed=42,
            number_components=8,
        )
        loader = tabular_timeseries_loader(args)
        train_loader, val_loader, test_loader, out_args = loader.load_dataset()

        # Loader tuple contract
        assert train_loader is not None and val_loader is not None and test_loader is not None
        assert len(train_loader.dataset) == 80
        assert len(val_loader.dataset) == 10
        assert len(test_loader.dataset) == 10

        # Args runtime fields
        assert out_args.seq_len == 10
        assert out_args.feat_dim == 1
        assert out_args.input_size == [1, 10]
        assert out_args.input_type == 'continuous'
        assert out_args.training_set_size == 80
        assert out_args.dynamic_binarization is False

    def test_train_loader_yields_3_tuple(self, tmp_path):
        """base_load_data.post_processing wraps train data with sample indices."""
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _build_csv_dataset(path, n_rows=64, n_t=8)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$')
        loader = tabular_timeseries_loader(args)
        train_loader, _, _, _ = loader.load_dataset()
        batch = next(iter(train_loader))
        assert len(batch) == 3, f"expected (x, idx, y), got {len(batch)}-tuple"

    def test_val_test_loaders_yield_2_tuple(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _build_csv_dataset(path, n_rows=64, n_t=8)
        args = _make_args(ts_path=str(path), ts_value_cols=r'^t_\d+$')
        loader = tabular_timeseries_loader(args)
        _, val_loader, test_loader, _ = loader.load_dataset()
        for ld in (val_loader, test_loader):
            batch = next(iter(ld))
            assert len(batch) == 2, f"expected (x, y), got {len(batch)}-tuple"

    def test_pre_split_mixed_formats(self, tmp_path):
        """train.csv + val.parquet + test.npy: each file resolved by own extension."""
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        train_path = tmp_path / "tr.csv"
        val_path = tmp_path / "va.parquet"
        test_path = tmp_path / "te.npy"
        # train as csv with t_0..t_4 + meta
        rng = np.random.default_rng(0)
        df_train = pd.DataFrame({
            **{f't_{i}': rng.random(20).astype(np.float32) for i in range(5)},
            'group': ['x'] * 20,
        })
        df_train.to_csv(train_path, index=False)
        # val as parquet, same value columns
        df_val = pd.DataFrame({
            **{f't_{i}': rng.random(5).astype(np.float32) for i in range(5)},
            'group': ['x'] * 5,
        })
        df_val.to_parquet(val_path)
        # test as npy: 5 rows × 5 timesteps
        np.save(test_path, rng.random((5, 5)).astype(np.float32))

        args = _make_args(
            ts_train_path=str(train_path),
            ts_val_path=str(val_path),
            ts_test_path=str(test_path),
            ts_value_cols=r'^t_\d+$',
        )
        loader = tabular_timeseries_loader(args)
        train_ld, val_ld, test_ld, out_args = loader.load_dataset()
        assert len(train_ld.dataset) == 20
        assert len(val_ld.dataset) == 5
        assert len(test_ld.dataset) == 5
        assert out_args.seq_len == 5

    def test_pre_split_T_mismatch(self, tmp_path):
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        train_path = tmp_path / "tr.csv"
        val_path = tmp_path / "va.csv"
        test_path = tmp_path / "te.csv"
        rng = np.random.default_rng(0)
        # train T=5
        pd.DataFrame({f't_{i}': rng.random(20) for i in range(5)}).to_csv(train_path, index=False)
        # val T=4 (mismatch!)
        pd.DataFrame({f't_{i}': rng.random(5) for i in range(4)}).to_csv(val_path, index=False)
        pd.DataFrame({f't_{i}': rng.random(5) for i in range(5)}).to_csv(test_path, index=False)

        args = _make_args(
            ts_train_path=str(train_path), ts_val_path=str(val_path),
            ts_test_path=str(test_path), ts_value_cols=r'^t_\d+$',
        )
        loader = tabular_timeseries_loader(args)
        with pytest.raises(SystemExit):
            loader.load_dataset()

    def test_label_propagates_to_loaders(self, tmp_path):
        """--ts_label_col actually flows through to the y tensors of the loaders."""
        from utils.load_data.timeseries_loader import tabular_timeseries_loader
        path = tmp_path / "d.csv"
        _build_csv_dataset(path, n_rows=60, n_t=5)  # group alternates A/B (mod 3)
        args = _make_args(
            ts_path=str(path),
            ts_value_cols=r'^t_\d+$',
            ts_label_col='group',
        )
        loader = tabular_timeseries_loader(args)
        train_ld, _, _, _ = loader.load_dataset()
        # collect all train labels
        labels = []
        for batch in train_ld:
            labels.extend(batch[-1].numpy().tolist())
        assert set(labels) == {0, 1}, "expected two distinct factorised labels"
