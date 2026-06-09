"""
Tests for fecg_open.core.mbeats_modify.mbeats_modify.

Key source behaviours verified:
  - Beat indices are 1-based throughout; xf is accessed as x_up[beat - 1].
  - SearchLen = 48 for both R-peak (max) and S-point (min) searches.
  - The output beat index is: mlocsf[ii] - 48 + argmax/argmin(window).
  - Out-of-bounds detections (index <= 0 or index >= len(signal)) are
    filtered.  Note the strict upper bound: index must be < len(signal),
    not <= len(signal), so the last sample is excluded.
  - Polarity correction: if |median(x_up[mbeats_q - 1])| >
    |median(x_up[mbeats_p - 1])|, swap p/q and return Po = -1.
  - R_amp = median(possibly-negated signal at mbeats_p - 1)
  - S_amp = median(possibly-negated signal at mbeats_q - 1)
"""

import warnings

import numpy as np
import pytest

from fecg_open.core.mbeats_modify import mbeats_modify

SEARCH_LEN = 48  # hard-coded in source


def _spike_signal(length: int, spike_pos: int, spike_val: float = 10.0) -> np.ndarray:
    """1-D signal of zeros with a single spike at spike_pos (1-based)."""
    x = np.zeros(length)
    x[spike_pos - 1] = spike_val
    return x


# ---------------------------------------------------------------------------
# 1. Exact peak alignment
# ---------------------------------------------------------------------------

def test_snaps_to_exact_peak():
    """
    Signal has a clear positive spike at position 100 (1-based).
    Approximate beat is 5 samples off (at 105).
    The function should return mbeats_p = [100] exactly.
    """
    N = 300
    spike_pos = 100
    approx_beat = 105   # 5 samples off

    x = _spike_signal(N, spike_pos, spike_val=5.0)
    mbeats_p, mbeats_q, R_amp, S_amp, Po = mbeats_modify(x, np.array([approx_beat]))

    np.testing.assert_array_equal(mbeats_p, [100])
    assert Po == 1


def test_snaps_within_search_window():
    """Offset within ±SearchLen is corrected; offset beyond is not."""
    N = 500
    spike_pos = 250

    # Offset of exactly SearchLen samples → still within window
    approx_beat = spike_pos + SEARCH_LEN
    x = _spike_signal(N, spike_pos, spike_val=8.0)
    mbeats_p, _, _, _, Po = mbeats_modify(x, np.array([approx_beat]))

    assert spike_pos in mbeats_p
    assert Po == 1


# ---------------------------------------------------------------------------
# 2. Polarity correction
# ---------------------------------------------------------------------------

def test_polarity_correction_inverted_signal():
    """
    Signal has a deep negative trough at position 100 and no positive peak.
    After search, the 'local max' is 0 (all zeros) and 'local min' is -10.
    |min| > |max| → Po = -1 and mbeats_p (R-peaks) snap to the trough.
    """
    N = 300
    trough_pos = 100
    x = np.zeros(N)
    x[trough_pos - 1] = -10.0          # deep trough, everything else 0

    approx_beat = trough_pos
    mbeats_p, mbeats_q, R_amp, S_amp, Po = mbeats_modify(x, np.array([approx_beat]))

    assert Po == -1, f"Expected Po=-1 for inverted signal, got {Po}"
    # After negation in source, R_amp should be positive
    assert R_amp > 0, f"R_amp should be positive after polarity flip, got {R_amp}"


def test_polarity_upright_signal():
    """A clearly positive spike → Po = +1."""
    N = 300
    x = _spike_signal(N, 150, spike_val=7.0)
    _, _, _, _, Po = mbeats_modify(x, np.array([150]))
    assert Po == 1


# ---------------------------------------------------------------------------
# 3. Out-of-bounds beats are filtered
# ---------------------------------------------------------------------------

def test_beat_near_start_filtered():
    """
    With a beat at mlocsf=1 and SearchLen=48, the computed R-peak index
    is 1 - 48 + argmax ≤ 0 for any argmax < 48.  Since the spike is at
    position 50 (outside the window starting from position 1 in a short
    signal), the window contains zeros → argmax=0 → mbeats_p_val = -47.
    Filtered out.  Output should be empty.
    """
    N = 30
    x = _spike_signal(N, 15, spike_val=3.0)   # spike at 15, window is [1, 29]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.errstate(invalid='ignore', divide='ignore'):
            mbeats_p, mbeats_q, R_amp, S_amp, Po = mbeats_modify(x, np.array([1]))

    # With spike at 15 in [1,29]: idx=14, mbeats_p_val = 1-48+14 = -33 → filtered
    assert len(mbeats_p) == 0, f"Expected empty mbeats_p, got {mbeats_p}"
    assert len(mbeats_q) == 0, f"Expected empty mbeats_q, got {mbeats_q}"


def test_beat_near_end_filtered():
    """
    The OOB filter uses a strict upper bound: mbeats_p < len(signal).
    A spike at the very last sample (1-based = N) produces mbeats_p_val = N,
    which fails the condition mbeats_p < N, and is therefore filtered out.

    This contrasts with a spike at N-1 (1-based), which produces
    mbeats_p_val = N-1 < N and is NOT filtered.
    """
    N = 300
    # Spike at the last sample (1-based = N = 300, 0-indexed = 299)
    x = _spike_signal(N, N, spike_val=5.0)
    approx_beat = N  # 1-based, points to the last sample

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with np.errstate(invalid='ignore', divide='ignore'):
            mbeats_p, mbeats_q, R_amp, S_amp, Po = mbeats_modify(
                x, np.array([approx_beat])
            )

    # mbeats_p_val = N - 48 + 48 = N = len(signal) → strict bound fails → filtered
    assert len(mbeats_p) == 0, f"Expected empty mbeats_p for last-sample spike, got {mbeats_p}"
    assert len(mbeats_q) == 0, f"Expected empty mbeats_q for last-sample spike, got {mbeats_q}"


def test_valid_beat_in_range_not_filtered():
    """A beat well within the signal is NOT filtered."""
    N = 500
    x = _spike_signal(N, 250, spike_val=4.0)
    mbeats_p, mbeats_q, _, _, _ = mbeats_modify(x, np.array([250]))
    assert len(mbeats_p) >= 1
    # All returned indices must be in the valid range (0, len(signal))
    assert np.all(mbeats_p > 0) and np.all(mbeats_p < N)
    assert np.all(mbeats_q > 0) and np.all(mbeats_q < N)


# ---------------------------------------------------------------------------
# 4. R_amp and S_amp match the median of x at the returned beat positions
# ---------------------------------------------------------------------------

def test_ramp_matches_median_at_mbeats_p():
    """
    R_amp returned by mbeats_modify must equal
    median(signal_used_internally[mbeats_p - 1]).

    When Po=+1 the internal signal equals the input x_up.
    When Po=-1 the internal signal equals -x_up (polarity-flipped).

    Signal parameters (spike at 250 in a 500-sample signal, approx beat at 250)
    guarantee a valid detection: mbeats_p_val = 250, 0 < 250 < 500.
    """
    N = 500
    spike_pos = 250
    x = _spike_signal(N, spike_pos, spike_val=6.0)

    mbeats_p, mbeats_q, R_amp, S_amp, Po = mbeats_modify(x, np.array([spike_pos]))

    assert len(mbeats_p) > 0, "No valid beats detected — check signal parameters"

    x_internal = Po * x
    expected_R = float(np.median(x_internal[mbeats_p - 1]))
    expected_S = float(np.median(x_internal[mbeats_q - 1]))

    assert R_amp == pytest.approx(expected_R, abs=1e-10)
    assert S_amp == pytest.approx(expected_S, abs=1e-10)


def test_ramp_samp_for_multiple_beats():
    """
    Verify R_amp / S_amp consistency with multiple beats.

    Signal parameters (spikes at 200 and 600 in a 1000-sample signal)
    guarantee valid detections: both mbeats_p values are in (0, 1000).
    """
    N = 1000
    x = np.zeros(N)
    x[199] = 5.0   # spike at 1-based 200
    x[599] = 7.0   # spike at 1-based 600

    mbeats_p, mbeats_q, R_amp, S_amp, Po = mbeats_modify(x, np.array([200, 600]))

    assert len(mbeats_p) > 0, "No valid beats detected — check signal parameters"

    x_internal = Po * x
    assert R_amp == pytest.approx(float(np.median(x_internal[mbeats_p - 1])), abs=1e-10)
    assert S_amp == pytest.approx(float(np.median(x_internal[mbeats_q - 1])), abs=1e-10)


# ---------------------------------------------------------------------------
# 5. Overlapping search windows with a close triplet of approximate beats
# ---------------------------------------------------------------------------

def test_overlapping_windows_all_snap_to_same_peak():
    """
    When three approximate beats [100, 110, 120] all lie within ±SearchLen
    of a single spike at position 100, every search window finds the same
    spike.  mbeats_modify processes each beat independently and does NOT
    deduplicate: the output contains three entries all equal to 100.
    """
    N = 400
    x = _spike_signal(N, 100, spike_val=8.0)

    mbeats_p, mbeats_q, _, _, Po = mbeats_modify(x, np.array([100, 110, 120]))

    # All three snap to the dominant peak
    np.testing.assert_array_equal(mbeats_p, [100, 100, 100])
    assert len(mbeats_p) == 3, "mbeats_modify must not deduplicate — three entries expected"
    assert Po == 1
