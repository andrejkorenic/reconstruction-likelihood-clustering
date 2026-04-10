"""Tests for pipeline flag dispatch in analyze.py and run.py."""
import pytest
import subprocess
import sys


class TestAnalyzeEntryPoint:
    def test_help_runs(self):
        """analyze.py --help exits cleanly with usage info."""
        result = subprocess.run(
            [sys.executable, 'analyze.py', '--help'],
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert '--cluster' in result.stdout
        assert '--recon_viz' in result.stdout
        assert '--generate' in result.stdout
        assert '--KNN' in result.stdout
        assert '--ood_scores' in result.stdout

    def test_no_flags_shows_error(self):
        """analyze.py --dir <path> without analysis flags exits with error."""
        result = subprocess.run(
            [sys.executable, 'analyze.py', '--dir', '/nonexistent/path/'],
            capture_output=True, text=True, timeout=10)
        assert result.returncode != 0

    def test_missing_checkpoint_shows_error(self):
        """analyze.py --dir <no-checkpoint> --cluster exits with error."""
        result = subprocess.run(
            [sys.executable, 'analyze.py', '--dir', '/tmp/', '--cluster'],
            capture_output=True, text=True, timeout=10)
        assert result.returncode != 0
        assert 'checkpoint_best.pth' in result.stdout or 'not found' in result.stdout

    def test_removed_dead_flags(self):
        """Dead flags from analysis.py are NOT in analyze.py."""
        result = subprocess.run(
            [sys.executable, 'analyze.py', '--help'],
            capture_output=True, text=True, timeout=10)
        assert '--just_log_likelihood' not in result.stdout
        assert '--count_active_dimensions' not in result.stdout
        assert '--grid_interpolation' not in result.stdout
        assert '--tsne_visualization' not in result.stdout
        assert '--hyper_lambda' not in result.stdout
        assert '--hidden_units' not in result.stdout


class TestRunPyFlags:
    def test_run_py_has_recon_viz(self):
        """run.py --help shows --recon_viz flag."""
        result = subprocess.run(
            [sys.executable, 'run.py', '--help'],
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert '--recon_viz' in result.stdout

    def test_run_py_has_generate(self):
        """run.py --help shows --generate flag."""
        result = subprocess.run(
            [sys.executable, 'run.py', '--help'],
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert '--generate' in result.stdout

    def test_run_py_has_cluster(self):
        """run.py --help shows --cluster flag (already existed)."""
        result = subprocess.run(
            [sys.executable, 'run.py', '--help'],
            capture_output=True, text=True, timeout=10)
        assert result.returncode == 0
        assert '--cluster' in result.stdout
