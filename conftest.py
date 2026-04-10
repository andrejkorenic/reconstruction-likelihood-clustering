"""Pytest configuration: --run-slow flag for slow integration tests."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="Run tests marked @pytest.mark.slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return  # run everything
    skip_slow = pytest.mark.skip(reason="need --run-slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
