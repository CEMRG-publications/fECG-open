import sys
import os
import pathlib

import numpy as np
import pytest

# Ensure the python/ directory is on sys.path so that
# `from fecg_open.core...` imports work when pytest is
# invoked from the repo root without a prior `pip install`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def rng():
    """Session-scoped NumPy random generator with fixed seed 42."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def test_recording_path():
    """Absolute path to the pre-trimmed 2-minute test recording directory."""
    return pathlib.Path(__file__).parent / "test_recording"
