"""
Tests for fecg_open.core.ECG_detrend.ECG_detrend.

Pipeline defaults used throughout: num_med_s=51, num_med_l=301.

Key source behaviours verified:
  - Output shape always equals input shape (both 1-D and 2-D).
  - A constant (DC) signal is detrended to near-zero for both modes.
  - A very slow sinusoid (0.1 Hz at 1000 Hz) has its amplitude substantially
    reduced (the short median window tracks it closely).
  - if_morph=0 (single median, aggressive) and if_morph=1 (cascaded, gentle)
    produce different outputs for the same input.
  - 2-D input is accepted and the output shape matches.
  - High-frequency content is preserved (not removed) by detrending.
"""

import numpy as np
import pytest

from fecg_open.core.ECG_detrend import ECG_detrend

# Pipeline defaults
NUM_MED_S = 51
NUM_MED_L = 301


# ---------------------------------------------------------------------------
# 1. Output length / shape matches input for both modes
# ---------------------------------------------------------------------------

def test_output_length_1d_morph0(rng):
    """Output shape matches input shape for a 1-D signal with morph=0."""
    x = rng.standard_normal(5000)
    result = ECG_detrend(x, NUM_MED_S, NUM_MED_L, 0)
    assert result.shape == x.shape


def test_output_length_1d_morph1(rng):
    """Output shape matches input shape for a 1-D signal with morph=1."""
    x = rng.standard_normal(5000)
    result = ECG_detrend(x, NUM_MED_S, NUM_MED_L, 1)
    assert result.shape == x.shape


# ---------------------------------------------------------------------------
# 2. DC signal → output near zero
#    Covers both morph modes and both DC=0 and DC≠0.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("if_morph,dc_val", [
    (0, 7.5),
    (1, 7.5),
    (0, 0.0),
    (1, 0.0),
])
def test_dc_signal_detrended_to_zero(if_morph, dc_val):
    """A constant (DC) signal is detrended to near-zero for both morphology modes and DC values."""
    x = np.full(1000, dc_val)
    result = ECG_detrend(x, NUM_MED_S, NUM_MED_L, if_morph)
    np.testing.assert_allclose(result, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 3. Slow sinusoid: amplitude substantially reduced (> 90%)
#
#    A 0.1 Hz sinusoid has period = 10 000 samples.  Both median windows
#    (51 and 301 samples) are << the period, so they track the slowly
#    varying baseline closely and the detrended residual is very small.
# ---------------------------------------------------------------------------

def _slow_sinusoid(N: int = 5000, fs: float = 1000.0, freq: float = 0.1) -> np.ndarray:
    """Return a unit-amplitude sinusoid at freq Hz, sampled at fs Hz, of length N samples."""
    t = np.arange(N) / fs
    return np.sin(2 * np.pi * freq * t)


def test_slow_sinusoid_reduced_morph0():
    """A 0.1 Hz sinusoid has more than 90 % of its RMS removed by morph=0 detrending."""
    x = _slow_sinusoid()
    result = ECG_detrend(x, NUM_MED_S, NUM_MED_L, 0)
    input_rms  = np.sqrt(np.mean(x ** 2))
    output_rms = np.sqrt(np.mean(result ** 2))
    assert output_rms < 0.10 * input_rms, (
        f"morph=0: residual RMS {output_rms:.4f} not < 10 % of input RMS {input_rms:.4f}"
    )


def test_slow_sinusoid_reduced_morph1():
    """A 0.1 Hz sinusoid has more than 90 % of its RMS removed by morph=1 detrending."""
    x = _slow_sinusoid()
    result = ECG_detrend(x, NUM_MED_S, NUM_MED_L, 1)
    input_rms  = np.sqrt(np.mean(x ** 2))
    output_rms = np.sqrt(np.mean(result ** 2))
    assert output_rms < 0.10 * input_rms, (
        f"morph=1: residual RMS {output_rms:.4f} not < 10 % of input RMS {input_rms:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. if_morph=0 vs if_morph=1: outputs must differ
# ---------------------------------------------------------------------------

def test_morph_modes_produce_different_outputs():
    """morph=0 and morph=1 produce numerically distinct outputs for the same input."""
    x = _slow_sinusoid()
    result_0 = ECG_detrend(x, NUM_MED_S, NUM_MED_L, 0)
    result_1 = ECG_detrend(x, NUM_MED_S, NUM_MED_L, 1)
    assert not np.allclose(result_0, result_1, atol=1e-6), (
        "if_morph=0 and if_morph=1 should produce different outputs"
    )


def test_morph0_not_less_aggressive_than_morph1_for_slow_wander():
    x = _slow_sinusoid()
    rms_0 = np.sqrt(np.mean(ECG_detrend(x, NUM_MED_S, NUM_MED_L, 0) ** 2))
    rms_1 = np.sqrt(np.mean(ECG_detrend(x, NUM_MED_S, NUM_MED_L, 1) ** 2))
    # Verified against MATLAB original: RMS morph=0 (~0.003) << RMS morph=1 (~0.014)
    assert rms_0 <= rms_1, (
        f"Expected morph=0 RMS ({rms_0:.5f}) <= morph=1 RMS ({rms_1:.5f})"
    )


# ---------------------------------------------------------------------------
# 5. 2-D input: accepted and output shape matches input shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("if_morph", [0, 1])
def test_2d_input_preserves_shape(rng, if_morph):
    """A 2-D input array is accepted and the output shape matches the input shape."""
    x_2d = rng.standard_normal((2, 3000))
    result = ECG_detrend(x_2d, NUM_MED_S, NUM_MED_L, if_morph)
    assert result.shape == x_2d.shape, (
        f"if_morph={if_morph}: expected shape {x_2d.shape}, got {result.shape}"
    )


def test_2d_dc_row_detrended():
    """2-D constant array should be detrended to near-zero."""
    x_2d = np.full((1, 1000), 3.0)
    result = ECG_detrend(x_2d, NUM_MED_S, NUM_MED_L, 0)
    np.testing.assert_allclose(result, 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 6. High-frequency content is preserved
#
#    The detrending removes slow baseline wander using median windows of
#    length 51 and 301 samples.  A 100 Hz sinusoid (period = 10 samples)
#    has period << both windows.  The sliding median of a zero-mean signal
#    evaluated over several complete periods is close to zero, so the
#    trend estimate for high-frequency content is near zero and the signal
#    passes through the subtraction step nearly intact.
# ---------------------------------------------------------------------------

def test_high_frequency_preserved_morph0():
    """100 Hz sinusoid passes through the morph=0 detrend path with >90% energy retained."""
    N = 5000
    fs = 1000.0
    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * 100.0 * t)
    result = ECG_detrend(x, NUM_MED_S, NUM_MED_L, 0)
    input_rms  = np.sqrt(np.mean(x ** 2))
    output_rms = np.sqrt(np.mean(result ** 2))
    assert output_rms > 0.90 * input_rms, (
        f"morph=0: high-freq RMS {output_rms:.4f} < 90 % of input RMS {input_rms:.4f}"
    )


def test_high_frequency_preserved_morph1():
    """100 Hz sinusoid passes through the morph=1 detrend path with >90% energy retained."""
    N = 5000
    fs = 1000.0
    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * 100.0 * t)
    result = ECG_detrend(x, NUM_MED_S, NUM_MED_L, 1)
    input_rms  = np.sqrt(np.mean(x ** 2))
    output_rms = np.sqrt(np.mean(result ** 2))
    assert output_rms > 0.90 * input_rms, (
        f"morph=1: high-freq RMS {output_rms:.4f} < 90 % of input RMS {input_rms:.4f}"
    )
