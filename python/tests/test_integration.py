"""
Integration test: run pipeline.py on the pre-trimmed test recording.

The test recording lives in ./python/tests/test_recording/ as four binary
files (test.ch1–test.ch4).  It is already a 2-minute excerpt and contains
exactly 2 segments — no runtime slicing is performed.

Output filename derivation (verified from pipeline.py):
    DB_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.ch1')])
    file_name = db_file.replace('.ch1', '')          # → 'test'
    output_file_path = os.path.join(output_folder, file_name + '.mat')  # → 'test.mat'

Expected HDF5 keys:
    Read directly from the mdict in parsave_output_file() — 11 keys, written
    by every call including the initial empty-file write before the segment loop.
"""

import pathlib
import subprocess
import sys

import h5py
import numpy as np
import pytest

# Repo root: this file lives at python/tests/test_integration.py
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Keys written by parsave_output_file() — taken verbatim from its mdict.
EXPECTED_KEYS = {
    "final_file_names",
    "final_start_times",
    "final_og_ECG",
    "final_SQI",
    "final_chs_used",
    "final_Om",
    "final_Of",
    "final_aECG",
    "final_mbeats",
    "final_fbeats",
    "final_proc_time",
}


def test_pipeline_real_recording(test_recording_path, tmp_path):
    """
    Run pipeline.py on the pre-trimmed test recording and verify output.

    Asserts (in order):
      - pipeline exits with returncode 0
      - test.mat exists in tmp_path
      - all 11 expected HDF5 keys are present
      - final_Om, final_Of, final_aECG entry counts are equal and > 0
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "python" / "pipeline.py"),
            "-i", str(test_recording_path),
            "-o", str(output_dir),
        ],
        capture_output=True,
        text=True,
        # 30-minute ceiling: 2 real segments × ~84 alpha/pair combinations
        # each in the sequential pipeline.
        timeout=1800,
        cwd=str(_REPO_ROOT),
    )

    if result.returncode != 0:
        pytest.fail(
            f"pipeline.py exited with code {result.returncode}.\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-4000:]}"
        )

    # parsave_output_file() creates the output file before the segment loop,
    # so test.mat must exist regardless of whether any segment succeeded.
    mat_path = output_dir / "test.mat"
    assert mat_path.exists(), (
        f"Expected output file {mat_path} not found.\n"
        f"Files in output_dir: {list(output_dir.iterdir())}\n"
        f"pipeline stdout:\n{result.stdout[-2000:]}"
    )

    with h5py.File(mat_path, "r") as f:
        missing = EXPECTED_KEYS - set(f.keys())
        assert not missing, f"Missing keys in output .mat: {missing}"

        def _n_entries(key):
            return int(np.asarray(f[key]).size)

        n_om  = _n_entries("final_Om")
        n_of  = _n_entries("final_Of")
        n_ecg = _n_entries("final_aECG")

        assert n_om == n_of == n_ecg, (
            f"Entry-count mismatch: final_Om={n_om}, "
            f"final_Of={n_of}, final_aECG={n_ecg}"
        )

        assert n_om > 0, (
            "No segments were written by the pipeline. "
            "The test recording should produce at least one valid segment.\n"
            f"pipeline stdout:\n{result.stdout[-2000:]}"
        )
