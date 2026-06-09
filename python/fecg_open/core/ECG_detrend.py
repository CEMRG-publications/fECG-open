"""
ECG_detrend.py: Cascaded-median baseline wander removal for ECG signals.

Remove slow baseline drift from an ECG signal using a cascaded median-filter approach,
optionally preserving beat morphology.

INPUTS:
    x0        — 1×N signal, raw ECG
    num_med_s — short window length (samples); 51 (no-morph) or 101 (morph)
    num_med_l — long window length (samples); 301 in main
    if_morph  — 0/1; 1 = cascaded median (gentle, preserves beat shape);
                       0 = single median (aggressive, removes morphology)

OUTPUTS:
    x0 — 1×N baseline-removed ECG signal

METHOD:
    if_morph=1: short movmedian → long movmedian → loess smooth (span=10) → subtract
    if_morph=0: short movmedian → loess smooth (span=10) → subtract

Uses scipy.ndimage.median_filter for the interior with per-edge fixup to replicate
MATLAB's truncated-window boundary behaviour exactly.
"""

import numpy as np
from scipy.ndimage import median_filter as _scipy_medfilt
from ..utils.matlab_smooth_wrapper import matlab_smooth


def _movmedian(x, k):
    """
    Moving median matching MATLAB's movmedian(x, k).

    MATLAB uses a centred window of length k; at boundaries it truncates to
    the samples actually available rather than padding. scipy's median_filter
    pads with reflected values, so boundary values diverge from MATLAB.
    This implementation uses the fast C-level scipy kernel for the interior
    and recomputes edge windows individually to match MATLAB exactly.
    """
    n = len(x)
    half_k = k // 2

    # Interior: scipy median filter (C-optimized, ~100x faster than Python loop)
    # mode='reflect' is close but not exact at edges — we fix edges below
    y = _scipy_medfilt(x, size=k, mode='reflect')

    # Fix edges where MATLAB uses truncated (shorter) windows
    # Left edge: positions 0 to half_k-1
    for i in range(min(half_k, n)):
        start = max(0, i - half_k)
        end = min(n, i + half_k + 1)
        y[i] = np.median(x[start:end])

    # Right edge: positions n-half_k to n-1
    for i in range(max(0, n - half_k), n):
        start = max(0, i - half_k)
        end = min(n, i + half_k + 1)
        y[i] = np.median(x[start:end])

    return y


def ECG_detrend(x0, num_med_s, num_med_l, if_morph):
    """
    Remove baseline wander from an ECG signal.

    Parameters
    ----------
    x0 : array-like
        1×N raw ECG signal.
    num_med_s : int
        Short window length (samples); 51 (no-morph) or 101 (morph).
    num_med_l : int
        Long window length (samples); 301 in main.
    if_morph : int
        0 = single median (aggressive, removes morphology);
        1 = cascaded median (gentle, preserves beat shape).

    Returns
    -------
    x0_detrended : ndarray
        1×N baseline-removed ECG signal.
    """
    x0 = np.asarray(x0, dtype=float)
    original_shape = x0.shape
    x0_flat = x0.flatten()

    if if_morph:
        x0_trend = _movmedian(x0_flat, num_med_s)
        x0_trend = _movmedian(x0_trend, num_med_l)
    else:
        x0_trend = _movmedian(x0_flat, num_med_s)

    x0_trend = matlab_smooth(x0_trend, 10)

    x0_detrended = x0_flat - x0_trend

    if len(original_shape) > 1:
        x0_detrended = x0_detrended.reshape(original_shape)

    return x0_detrended
