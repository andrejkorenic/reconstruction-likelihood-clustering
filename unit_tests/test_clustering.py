"""
Unit tests for clustering metrics return structure and CSV output.
Uses synthetic data — does not require a trained model.
"""
import csv
from pathlib import Path
import numpy as np
import pytest

from utils.clustering import (
    _evaluate_and_log_unsupervised,
    _evaluate_and_log_supervised,
    _perform_and_evaluate_clustering,
    _ALL_CLUSTERING_ALGORITHMS,
    save_metrics_csv,
    _METRIC_COLUMNS,
)

@pytest.fixture
def synthetic_data():
    """10 clusters of 50 points each in 40-D space, well-separated."""
    rng = np.random.RandomState(42)
    centers = rng.randn(10, 40) * 10
    data = np.vstack([centers[i] + rng.randn(50, 40) * 0.5 for i in range(10)])
    labels = np.repeat(np.arange(10), 50)
    return data, labels

class TestEvaluateUnsupervised:
    def test_returns_dict_with_label_metrics(self, synthetic_data):
        data, labels = synthetic_data
        result = _evaluate_and_log_unsupervised(data, labels, labels, "test", has_labels=True)
        assert isinstance(result, dict)
        for key in ("ARI", "AMI", "SS", "DBI", "CHI"):
            assert key in result
            assert isinstance(result[key], float)

    def test_returns_dict_without_label_metrics(self, synthetic_data):
        data, labels = synthetic_data
        result = _evaluate_and_log_unsupervised(data, labels, labels, "test", has_labels=False)
        assert "ARI" not in result
        assert "AMI" not in result
        for key in ("SS", "DBI", "CHI"):
            assert key in result

class TestEvaluateSupervised:
    def test_returns_dict_with_acc_fms_vms(self, synthetic_data):
        data, labels = synthetic_data
        result = _evaluate_and_log_supervised(data, labels, labels)
        assert isinstance(result, dict)
        for key in ("ACC", "FMS", "VMS"):
            assert key in result
            assert isinstance(result[key], float)

    def test_perfect_labels_give_acc_1(self, synthetic_data):
        data, labels = synthetic_data
        result = _evaluate_and_log_supervised(data, labels, labels)
        assert result["ACC"] == pytest.approx(1.0)

class TestPerformClustering:
    def test_returns_list_of_dicts(self, synthetic_data):
        data, labels = synthetic_data
        results = _perform_and_evaluate_clustering(data, labels, "raw")
        assert isinstance(results, list)
        assert len(results) == 3  # knn, kmeans, hdbscan

    def test_each_row_has_required_keys(self, synthetic_data):
        data, labels = synthetic_data
        results = _perform_and_evaluate_clustering(data, labels, "raw")
        for row in results:
            assert "embedding" in row
            assert "method" in row
            assert row["embedding"] == "raw"
            assert row["method"] in ("knn", "kmeans", "hdbscan")

    def test_hdbscan_has_pct_clustered(self, synthetic_data):
        data, labels = synthetic_data
        results = _perform_and_evaluate_clustering(data, labels, "raw")
        hdbscan_rows = [r for r in results if r["method"] == "hdbscan"]
        assert len(hdbscan_rows) == 1
        assert hdbscan_rows[0]["pct_clustered"] is not None
        assert isinstance(hdbscan_rows[0]["pct_clustered"], float)

    def test_non_hdbscan_pct_clustered_is_none(self, synthetic_data):
        data, labels = synthetic_data
        results = _perform_and_evaluate_clustering(data, labels, "raw")
        non_hdbscan = [r for r in results if r["method"] != "hdbscan"]
        for row in non_hdbscan:
            assert row["pct_clustered"] is None

    def test_well_separated_clusters_give_high_ari(self, synthetic_data):
        data, labels = synthetic_data
        results = _perform_and_evaluate_clustering(data, labels, "raw")
        knn_row = [r for r in results if r["method"] == "knn"][0]
        assert knn_row["ARI"] > 0.8

class TestSaveMetricsCsv:
    def test_creates_csv_file(self, tmp_path):
        results = [
            {"embedding": "raw", "method": "knn", "ARI": 0.9, "AMI": 0.85,
             "SS": 0.5, "DBI": 1.2, "CHI": 1500.0, "ACC": 0.95, "FMS": 0.9,
             "VMS": 0.85, "pct_clustered": None},
        ]
        out = tmp_path / "metrics.csv"
        save_metrics_csv(results, out)
        assert out.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        results = [
            {"embedding": "raw", "method": "knn", "ARI": 0.9, "AMI": 0.85,
             "SS": 0.5, "DBI": 1.2, "CHI": 1500.0, "ACC": 0.95, "FMS": 0.9,
             "VMS": 0.85, "pct_clustered": None},
        ]
        out = tmp_path / "metrics.csv"
        save_metrics_csv(results, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == _METRIC_COLUMNS

    def test_csv_round_trip(self, tmp_path):
        results = [
            {"embedding": "tsne", "method": "hdbscan", "ARI": 0.1234, "AMI": 0.5678,
             "SS": 0.3, "DBI": 2.5, "CHI": 800.0, "ACC": 0.75, "FMS": 0.6,
             "VMS": 0.55, "pct_clustered": 85.3},
        ]
        out = tmp_path / "metrics.csv"
        save_metrics_csv(results, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["embedding"] == "tsne"
        assert rows[0]["method"] == "hdbscan"
        assert float(rows[0]["ARI"]) == pytest.approx(0.1234, abs=1e-4)
        assert float(rows[0]["pct_clustered"]) == pytest.approx(85.3, abs=0.1)

    def test_multiple_rows(self, tmp_path, synthetic_data):
        data, labels = synthetic_data
        results = _perform_and_evaluate_clustering(data, labels, "raw")
        out = tmp_path / "metrics.csv"
        save_metrics_csv(results, out)
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3  # knn, kmeans, hdbscan
