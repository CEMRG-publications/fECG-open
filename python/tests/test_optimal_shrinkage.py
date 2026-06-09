"""
Tests for:
  - fecg_open.core.optimal_shrinkage.optimal_shrinkage
  - fecg_open.core.ECG_shrinkage0.ECG_shrinkage0
  - fecg_open.core.ECG_shrinkage_median.ECG_shrinkage_median

optimal_shrinkage shrinkage function:
    eta(y) = sigma * max(x(y/sigma), 0)
    x(u)   = sqrt(0.5 * ((u^2 - beta - 1) + sqrt((u^2 - beta - 1)^2 - 4*beta)))
             if u >= 1 + sqrt(beta), else 0.

ECG_shrinkage0 and ECG_shrinkage_median build a beat matrix from a multi-beat
signal and apply optimal_shrinkage to its singular values, then reconstruct
via raised-cosine overlap-add.
"""

import numpy as np
import pytest

from fecg_open.core.optimal_shrinkage import optimal_shrinkage
from fecg_open.core.ECG_shrinkage0 import ECG_shrinkage0
from fecg_open.core.ECG_shrinkage_median import ECG_shrinkage_median


# ===========================================================================
# optimal_shrinkage tests
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. All values below threshold → all shrunk to zero
# ---------------------------------------------------------------------------

def test_all_below_threshold_zeroed():
    # beta=1, sigma=1: threshold = sigma*(1+sqrt(1)) = 2.0
    singvals = np.array([0.1, 0.5, 1.0, 1.9])   # all < 2.0
    result = optimal_shrinkage(singvals, beta=1.0, sigma=1.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Explicit sigma: hand-computable output for beta=1, sigma=1, y=2
# ---------------------------------------------------------------------------

def test_explicit_sigma_hand_computable_at_threshold():
    """
    beta=1, sigma=1, y=2 (exactly at threshold 1+sqrt(1)=2):
      u = y/sigma = 2
      term = (4-1-1)^2 - 4*1 = 4 - 4 = 0
      x(u) = sqrt(0.5 * (2 + 0)) = sqrt(1) = 1.0
      output = sigma * 1.0 = 1.0
    """
    result = optimal_shrinkage(np.array([2.0]), beta=1.0, sigma=1.0)
    assert result == pytest.approx([1.0], abs=1e-10)


def test_explicit_sigma_hand_computable_above_threshold():
    """
    beta=1, sigma=1, y=3:
      u = 3; term = (9-1-1)^2 - 4 = 49 - 4 = 45
      x(3) = sqrt(0.5 * (7 + sqrt(45)))
           = sqrt(0.5 * (7 + 6.7082...))
           = sqrt(6.8541...)
           ≈ 2.6181
    """
    expected = np.sqrt(0.5 * (7.0 + np.sqrt(45.0)))
    result = optimal_shrinkage(np.array([3.0]), beta=1.0, sigma=1.0)
    assert result == pytest.approx([expected], rel=1e-6)


def test_explicit_sigma_mixed_above_and_below():
    """Combines zeroed and non-zeroed entries in one call."""
    # beta=1, sigma=1: threshold=2; singvals=[1.5, 2.0, 3.0]
    result = optimal_shrinkage(np.array([1.5, 2.0, 3.0]), beta=1.0, sigma=1.0)
    expected_2 = np.sqrt(0.5 * (2.0 + np.sqrt(0.0)))          # 1.0
    expected_3 = np.sqrt(0.5 * (7.0 + np.sqrt(45.0)))         # ≈ 2.618
    assert result[0] == pytest.approx(0.0,         abs=1e-10)
    assert result[1] == pytest.approx(expected_2,  abs=1e-10)
    assert result[2] == pytest.approx(expected_3,  rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Sigma estimated from data: output is valid (non-negative, same length)
# ---------------------------------------------------------------------------

def test_sigma_estimated_output_is_valid():
    """Output with estimated sigma is non-negative, finite, and the same length as the input."""
    singvals = np.array([1.0, 3.0, 5.0, 10.0, 20.0])
    result = optimal_shrinkage(singvals, beta=0.5)
    assert len(result) == len(singvals), "output length must equal input length"
    assert np.all(result >= 0),          "all shrunk singular values must be non-negative"
    assert np.all(np.isfinite(result)),  "no NaN or Inf in output"


def test_sigma_estimated_monotone():
    """Larger singular values should not produce smaller shrunk values."""
    singvals = np.sort(np.array([0.5, 1.5, 3.0, 6.0, 12.0]))
    result = optimal_shrinkage(singvals, beta=0.3)
    assert np.all(np.diff(result) >= 0), "shrunk values should be non-decreasing"


# ---------------------------------------------------------------------------
# 4. beta=1: threshold is 2*sigma; values below zeroed, at/above non-zero
# ---------------------------------------------------------------------------

def test_beta1_threshold_is_two_sigma():
    sigma = 2.0
    # threshold = sigma * (1 + sqrt(1)) = 4.0
    singvals = np.array([1.0, 3.9, 4.0, 8.0])
    result = optimal_shrinkage(singvals, beta=1.0, sigma=sigma)

    assert result[0] == pytest.approx(0.0, abs=1e-10)   # 1.0 < 4.0 → zero
    assert result[1] == pytest.approx(0.0, abs=1e-10)   # 3.9 < 4.0 → zero

    # y=4.0 is exactly at threshold: u=4/2=2, same formula as test 2
    assert result[2] == pytest.approx(sigma * 1.0, abs=1e-10)  # sigma * x(2) = 2.0

    assert result[3] > 0                                 # 8.0 > 4.0 → positive


# ---------------------------------------------------------------------------
# 5. All-zero singular values → all-zero output
# ---------------------------------------------------------------------------

def test_allzero_input_returns_allzero():
    singvals = np.zeros(6)
    # With explicit sigma so no Marchenko-Pastur estimation needed
    result = optimal_shrinkage(singvals, beta=0.5, sigma=1.0)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Invalid beta raises AssertionError
# ---------------------------------------------------------------------------

def test_beta_zero_raises():
    """beta=0 raises AssertionError because beta must be strictly positive."""
    with pytest.raises(AssertionError):
        optimal_shrinkage(np.array([1.0, 2.0]), beta=0.0)


def test_beta_negative_raises():
    """A negative beta raises AssertionError."""
    with pytest.raises(AssertionError):
        optimal_shrinkage(np.array([1.0, 2.0]), beta=-0.5)


def test_beta_above_one_raises():
    """beta > 1 raises AssertionError because beta must be in (0, 1]."""
    with pytest.raises(AssertionError):
        optimal_shrinkage(np.array([1.0, 2.0]), beta=1.5)


# ===========================================================================
# Shared helpers for ECG_shrinkage0 / ECG_shrinkage_median tests
# ===========================================================================

def _make_periodic_ecg(n=2000, beat_spacing=150, beat_amp=5.0, noise_std=0.1, seed=42):
    """
    Synthetic ECG: Gaussian-spike beats at regular intervals plus low-level noise.
    Returns (signal_1d, beats_1based).

    With default parameters: 12 beats at positions 150, 300, ..., 1800.
    After boundary trimming inside ECG_shrinkage0/median (MaximalQTp = ceil(150*0.5) = 75):
    all beats satisfy beat > 75 and beat + 75 <= 2000, so all 12 survive.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) * noise_std
    beats = []
    pos = beat_spacing
    while pos + beat_spacing <= n:
        x[pos - 1] += beat_amp   # 1-based → 0-indexed
        beats.append(pos)
        pos += beat_spacing
    return x, np.array(beats, dtype=int)


# ===========================================================================
# ECG_shrinkage0 tests
# ===========================================================================

class TestECGShrinkage0:
    """
    Tests for ECG_shrinkage0(x0, x0_real, current_beats, sigma_coeff, ifauto).

    The function builds a beat matrix from windows centred on each R-peak,
    applies operator-norm SVD shrinkage, and reconstructs via overlap-add.
    """

    def test_output_shape(self):
        """Both outputs have the same length as the input signal."""
        x, beats = _make_periodic_ecg()
        Om0, Om0_real = ECG_shrinkage0(x, x.copy(), beats, sigma_coeff=1.0, ifauto=0)
        assert len(Om0) == len(x)
        assert len(Om0_real) == len(x)

    def test_output_finite(self):
        """No NaN or Inf in output for a well-conditioned input."""
        x, beats = _make_periodic_ecg()
        Om0, Om0_real = ECG_shrinkage0(x, x.copy(), beats, sigma_coeff=1.0, ifauto=0)
        assert np.all(np.isfinite(Om0)),      "Om0 contains non-finite values"
        assert np.all(np.isfinite(Om0_real)), "Om0_real contains non-finite values"

    def test_passthrough_when_no_beats_survive_boundary_trim(self):
        """
        When both beats are trimmed by boundary checks, the function returns
        the input arrays unchanged.

        Calculation: beats=[5, 1000], n=1100.
          RRI = [995]; MaximalQTp = ceil(995 * 4/8) = 498.
          Trim 1 (beats > 498): only beat 1000 survives.
          Trim 2 (beat + 498 <= 1100): 1498 > 1100 → beat 1000 also trimmed.
          → empty beat list → early return.
        """
        n = 1100
        x = np.arange(n, dtype=float)
        x_real = x * 2.0
        beats = np.array([5, 1000])
        Om0, Om0_real = ECG_shrinkage0(x, x_real, beats, sigma_coeff=1.0, ifauto=0)
        np.testing.assert_array_equal(Om0, x)
        np.testing.assert_array_equal(Om0_real, x_real)

    def test_beat_amplitude_retained(self):
        """
        For a periodic signal with low noise, the dominant singular value
        survives shrinkage and the output retains significant amplitude at
        beat positions.
        """
        beat_amp = 5.0
        x, beats = _make_periodic_ecg(beat_amp=beat_amp, noise_std=0.1)
        Om0, _ = ECG_shrinkage0(x, x.copy(), beats, sigma_coeff=1.0, ifauto=0)

        assert len(Om0) == len(x)
        assert np.all(np.isfinite(Om0))
        # Output at beat positions should retain at least half the original amplitude
        beat_output = np.mean(np.abs(Om0[beats - 1]))
        assert beat_output > 0.5 * beat_amp, (
            f"Output amplitude at beats ({beat_output:.3f}) < 50 % of input ({beat_amp})"
        )

    def test_ifauto_1_output_shape_and_finite(self):
        """ifauto=1 (auto noise estimation) also produces valid output."""
        x, beats = _make_periodic_ecg()
        Om0, Om0_real = ECG_shrinkage0(x, x.copy(), beats, sigma_coeff=1.0, ifauto=1)
        assert len(Om0) == len(x)
        assert len(Om0_real) == len(x)
        assert np.all(np.isfinite(Om0))
        assert np.all(np.isfinite(Om0_real))


# ===========================================================================
# ECG_shrinkage_median tests
# ===========================================================================

class TestECGShrinkageMedian:
    """
    Tests for ECG_shrinkage_median(x0, x0_real, current_beats,
                                    num_nonlocal, sigma_coeff, ifauto).

    Builds the same beat matrix as ECG_shrinkage0 (but using the 70th-percentile
    RRI window) then replaces each beat's waveform with the median of its
    num_nonlocal nearest RR-neighbours before overlap-add reconstruction.
    """

    def test_output_shape(self):
        """Both outputs have the same length as the input signal."""
        x, beats = _make_periodic_ecg()
        Om0, Om0_real = ECG_shrinkage_median(
            x, x.copy(), beats, num_nonlocal=3, sigma_coeff=1.0, ifauto=0
        )
        assert len(Om0) == len(x)
        assert len(Om0_real) == len(x)

    def test_output_finite(self):
        """No NaN or Inf in output for a well-conditioned input."""
        x, beats = _make_periodic_ecg()
        Om0, Om0_real = ECG_shrinkage_median(
            x, x.copy(), beats, num_nonlocal=3, sigma_coeff=1.0, ifauto=0
        )
        assert np.all(np.isfinite(Om0)),      "Om0 contains non-finite values"
        assert np.all(np.isfinite(Om0_real)), "Om0_real contains non-finite values"

    def test_passthrough_when_no_beats_survive_boundary_trim(self):
        """
        When both beats are trimmed by boundary checks, the function returns
        the input arrays unchanged.  Same boundary arithmetic as ECG_shrinkage0
        but uses 70th-percentile RRI window.

        With beats=[5, 1000] and n=1100, pctile_70 = 995 (constant RRI),
        MaximalQTp = ceil(995 * 4/8) = 498 → same trimming result as shrinkage0.
        """
        n = 1100
        x = np.arange(n, dtype=float)
        x_real = x * 2.0
        beats = np.array([5, 1000])
        Om0, Om0_real = ECG_shrinkage_median(
            x, x_real, beats, num_nonlocal=1, sigma_coeff=1.0, ifauto=0
        )
        np.testing.assert_array_equal(Om0, x)
        np.testing.assert_array_equal(Om0_real, x_real)

    def test_beat_amplitude_retained(self):
        """
        For a periodic signal with low noise, the non-local median reconstruction
        retains significant amplitude at beat positions.
        """
        beat_amp = 5.0
        x, beats = _make_periodic_ecg(beat_amp=beat_amp, noise_std=0.1)
        Om0, _ = ECG_shrinkage_median(
            x, x.copy(), beats, num_nonlocal=3, sigma_coeff=1.0, ifauto=0
        )

        assert len(Om0) == len(x)
        assert np.all(np.isfinite(Om0))
        beat_output = np.mean(np.abs(Om0[beats - 1]))
        assert beat_output > 0.5 * beat_amp, (
            f"Output amplitude at beats ({beat_output:.3f}) < 50 % of input ({beat_amp})"
        )

    def test_num_nonlocal_1_vs_3_both_valid(self):
        """
        num_nonlocal=1 (beat uses only itself) and num_nonlocal=3 (uses 3 RR
        neighbours) both produce finite outputs of the correct shape.  For a
        signal with identical RR intervals, the neighbourhood waveforms are
        similar so the two outputs should be numerically close.
        """
        x, beats = _make_periodic_ecg(noise_std=0.05)

        Om0_n1, _ = ECG_shrinkage_median(
            x, x.copy(), beats, num_nonlocal=1, sigma_coeff=1.0, ifauto=0
        )
        Om0_n3, _ = ECG_shrinkage_median(
            x, x.copy(), beats, num_nonlocal=3, sigma_coeff=1.0, ifauto=0
        )

        assert len(Om0_n1) == len(x)
        assert len(Om0_n3) == len(x)
        assert np.all(np.isfinite(Om0_n1))
        assert np.all(np.isfinite(Om0_n3))

    def test_ifauto_1_output_shape_and_finite(self):
        """ifauto=1 (auto noise estimation) also produces valid output."""
        x, beats = _make_periodic_ecg()
        Om0, Om0_real = ECG_shrinkage_median(
            x, x.copy(), beats, num_nonlocal=3, sigma_coeff=1.0, ifauto=1
        )
        assert len(Om0) == len(x)
        assert len(Om0_real) == len(x)
        assert np.all(np.isfinite(Om0))
        assert np.all(np.isfinite(Om0_real))
