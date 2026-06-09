"""
Tests for fecg_open.core.RRconstraint.RRconstraint.

Key source behaviours verified here:
  - Comparison is <=: pairs with RRI exactly equal to RS*fs ARE too close.
  - The beat with the smaller |xf| value is flagged for removal.
  - Removal is done all at once via np.setdiff1d (not iteratively).
  - Output is always sorted (setdiff1d guarantee).
  - Beats are 1-based; xf is accessed as xf[beat - 1].
"""

import numpy as np
import pytest

from fecg_open.core.RRconstraint import RRconstraint


# ---------------------------------------------------------------------------
# 1. Identity: all RRI above threshold → nothing removed
# ---------------------------------------------------------------------------

def test_identity_no_removal():
    fs = 1000
    RS = 0.25                         # threshold = 250 samples
    beats = np.array([100, 400, 700, 1000])   # all RRIs = 300 > 250
    xf = np.zeros(1100)
    for b in beats:
        xf[b - 1] = 1.0              # non-zero amplitude at each beat

    result = RRconstraint(beats, xf, fs, RS)
    np.testing.assert_array_equal(result, beats)


# ---------------------------------------------------------------------------
# 2. One too-close pair: the beat with smaller amplitude is removed
# ---------------------------------------------------------------------------

def test_one_pair_removes_smaller_amplitude():
    fs = 100
    RS = 0.25                         # threshold = 25 samples
    # beats 100 and 110 are 10 samples apart (< 25) → too close
    beats = np.array([100, 110, 200])
    xf = np.zeros(210)
    xf[99]  = 0.5   # beat 100 → smaller  → should be removed
    xf[109] = 1.5   # beat 110 → larger   → should be kept
    xf[199] = 1.0   # beat 200 → far from 110, always kept

    result = RRconstraint(beats, xf, fs, RS)
    np.testing.assert_array_equal(result, np.array([110, 200]))


def test_one_pair_removes_later_when_later_is_smaller():
    fs = 100
    RS = 0.25                         # threshold = 25 samples
    beats = np.array([100, 110, 200])
    xf = np.zeros(210)
    xf[99]  = 2.0   # beat 100 → larger  → kept
    xf[109] = 0.8   # beat 110 → smaller → removed
    xf[199] = 1.0

    result = RRconstraint(beats, xf, fs, RS)
    np.testing.assert_array_equal(result, np.array([100, 200]))


def test_threshold_is_inclusive():
    """RRI == RS*fs (exactly at threshold) should also trigger removal."""
    fs = 1000
    RS = 0.025                        # threshold = 25 samples exactly
    beats = np.array([100, 125, 500]) # RRI[0] = 25 == threshold → too close
    xf = np.zeros(600)
    xf[99]  = 0.5   # smaller → removed
    xf[124] = 1.5   # larger  → kept
    xf[499] = 1.0

    result = RRconstraint(beats, xf, fs, RS)
    np.testing.assert_array_equal(result, np.array([125, 500]))


# ---------------------------------------------------------------------------
# 3. Multiple independent too-close pairs, each resolved correctly
# ---------------------------------------------------------------------------

def test_multiple_independent_pairs():
    fs = 1000
    RS = 0.25                         # threshold = 250 samples
    # Two non-overlapping too-close pairs: (100, 110) and (500, 510)
    beats = np.array([100, 110, 500, 510])
    xf = np.zeros(600)
    xf[99]  = 0.5   # beat 100 → smaller in first pair  → removed
    xf[109] = 2.0   # beat 110 → larger in first pair   → kept
    xf[499] = 3.0   # beat 500 → larger in second pair  → kept
    xf[509] = 0.3   # beat 510 → smaller in second pair → removed

    result = RRconstraint(beats, xf, fs, RS)
    np.testing.assert_array_equal(result, np.array([110, 500]))


# ---------------------------------------------------------------------------
# 4. Chained / overlapping close pairs: batch-removal semantics
#
#    RRconstraint is a single-pass operation using np.setdiff1d.  When all
#    three consecutive beats [100, 110, 120] are too close, the function
#    evaluates both pairs (100,110) and (110,120) in one loop and builds
#    a flat list of beats to remove; setdiff1d then removes all flagged
#    beats at once.  The outcome depends on which beat carries the
#    smallest amplitude — the two cases below expose distinct semantics.
# ---------------------------------------------------------------------------

def test_chained_triplet_middle_beat_flagged_by_both_pairs():
    """
    When the middle beat has the smallest amplitude it is flagged from BOTH
    pairs.  setdiff1d removes it exactly once, leaving the two endpoint beats
    even though they are still only 20 samples apart (< 50-sample threshold).

    This demonstrates that RRconstraint is NOT iterative: a single pass is
    not sufficient to resolve cascaded violations.
    """
    fs = 1000
    RS = 0.050          # threshold = 50 samples; all gaps (10, 10) << 50
    beats = np.array([100, 110, 120])
    xf = np.zeros(200)
    xf[99]  = 3.0   # beat 100 → largest in pair (100,110)
    xf[109] = 1.0   # beat 110 → smallest in both pairs → flagged twice, removed once
    xf[119] = 2.0   # beat 120 → largest in pair (110,120)

    result = RRconstraint(beats, xf, fs, RS)

    # Pair(100,110): 3 > 1 → 110 flagged.  Pair(110,120): 1 < 2 → 110 flagged again.
    # fp = [110, 110]; setdiff1d([100,110,120], [110]) = [100, 120].
    # 100 and 120 are 20 samples apart — constraint still violated in the output.
    np.testing.assert_array_equal(result, np.array([100, 120]))


def test_chained_triplet_middle_beat_largest_both_endpoints_removed():
    """
    When the middle beat has the largest amplitude, one endpoint is flagged
    per pair.  setdiff1d removes both endpoints, leaving only the middle beat.
    This is the well-behaved case where a single pass fully resolves the
    triplet.
    """
    fs = 1000
    RS = 0.050
    beats = np.array([100, 110, 120])
    xf = np.zeros(200)
    xf[99]  = 1.0   # beat 100 → smaller in pair (100,110) → flagged
    xf[109] = 3.0   # beat 110 → largest overall → kept
    xf[119] = 2.0   # beat 120 → smaller in pair (110,120) → flagged

    result = RRconstraint(beats, xf, fs, RS)

    # Pair(100,110): 1 < 3 → 100 flagged.  Pair(110,120): 3 > 2 → 120 flagged.
    # fp = [100, 120]; setdiff1d([100,110,120], [100,120]) = [110].
    np.testing.assert_array_equal(result, np.array([110]))


# ---------------------------------------------------------------------------
# 5. Edge case: empty input → empty output
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty():
    """An empty beat array returns an empty array."""
    beats = np.array([], dtype=int)
    xf    = np.array([])
    result = RRconstraint(beats, xf, 1000, 0.25)
    assert result.shape == (0,)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# 6. Edge case: single beat → unchanged
# ---------------------------------------------------------------------------

def test_single_beat_unchanged():
    """A single-beat input is returned unchanged."""
    beats = np.array([100])
    xf = np.zeros(200)
    xf[99] = 1.5
    result = RRconstraint(beats, xf, 1000, 0.25)
    np.testing.assert_array_equal(result, beats)


# ---------------------------------------------------------------------------
# 7. RS = 0 → no refractory period → no beats removed regardless of spacing
# ---------------------------------------------------------------------------

def test_rs_zero_no_removal():
    """RRI[i] >= 1 for distinct sorted beats, so RRI[i] <= 0 is always False."""
    fs = 1000
    RS = 0
    beats = np.array([100, 101, 102, 103])  # consecutive — extremely close
    xf = np.zeros(110)
    for b in beats:
        xf[b - 1] = float(b)           # distinct amplitudes

    result = RRconstraint(beats, xf, fs, RS)
    np.testing.assert_array_equal(result, beats)
