"""
Unit tests for LL estimation pipeline: CSV output and flag logic.
Uses synthetic data where possible — does not require a trained model.
"""
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.evaluation import save_ll_metrics_csv, _LL_METRIC_COLUMNS


class TestSaveLlMetricsCsv:
    def _make_metrics(self, **overrides):
        defaults = {
            "test_nll": -84.45,
            "test_elbo": 88.04,
            "train_elbo": 91.79,
            "val_elbo": 88.11,
            "test_re": 61.43,
            "test_kl": 26.61,
            "S": 5000,
            "epochs_trained": 930,
        }
        defaults.update(overrides)
        return defaults

    def test_creates_csv_file(self, tmp_path):
        out = tmp_path / "ll_metrics.csv"
        save_ll_metrics_csv(self._make_metrics(), out)
        assert out.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        out = tmp_path / "ll_metrics.csv"
        save_ll_metrics_csv(self._make_metrics(), out)
        with open(out) as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == _LL_METRIC_COLUMNS

    def test_csv_round_trip_values(self, tmp_path):
        metrics = self._make_metrics()
        out = tmp_path / "ll_metrics.csv"
        save_ll_metrics_csv(metrics, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["test_nll"]) == pytest.approx(-84.45, abs=0.01)
        assert float(rows[0]["test_elbo"]) == pytest.approx(88.04, abs=0.01)
        assert int(rows[0]["S"]) == 5000
        assert int(rows[0]["epochs_trained"]) == 930

    def test_handles_none_test_nll(self, tmp_path):
        metrics = self._make_metrics(test_nll=None, S=None)
        out = tmp_path / "ll_metrics.csv"
        save_ll_metrics_csv(metrics, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["test_nll"] == ""
        assert rows[0]["S"] == ""
        assert float(rows[0]["test_elbo"]) == pytest.approx(88.04, abs=0.01)

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "ll_metrics.csv"
        save_ll_metrics_csv(self._make_metrics(), out)
        assert out.exists()


class TestLlFlagLogic:
    @staticmethod
    def _resolve_ll(ll_flag, has_dir):
        if ll_flag is None:
            return has_dir
        return ll_flag

    def test_train_mode_default_skips_ll(self):
        assert self._resolve_ll(ll_flag=None, has_dir=False) is False

    def test_load_mode_default_computes_ll(self):
        assert self._resolve_ll(ll_flag=None, has_dir=True) is True

    def test_explicit_ll_overrides_train_mode(self):
        assert self._resolve_ll(ll_flag=True, has_dir=False) is True

    def test_explicit_no_ll_overrides_load_mode(self):
        assert self._resolve_ll(ll_flag=False, has_dir=True) is False
