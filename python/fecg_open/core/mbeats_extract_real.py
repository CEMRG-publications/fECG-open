"""
mbeats_extract_real: Extract maternal ECG R-peaks from an abdominal mixture signal
using a time-frequency ridge-following and dynamic-programming beat tracker.

INPUTS:
    x0        — 1×N double, de-trended mixed ECG (for beat timing)
    x0_real   — 1×N double, morphology-preserving de-trended mixed ECG
    fs        — scalar, signal sampling rate (Hz), nominally 1000
    basicTF   — dict: STFT parameters (win, hop, fs=100, fr=0.02)
    advTF     — dict: SST thresholds (ths, HighFreq, LowFreq)
    cepR      — dict: cepstral parameters (g=0.3, Tc=0)
    P         — dict: P['num_s']=1
    lam_curve — scalar >=0, DP smoothness weight for HR ridge extraction
    lam_beat  — scalar >=0, DP transition weight for beat tracker

OUTPUTS:
    x0        — 1×N double, polarity-corrected version of x0
    x0_real   — 1×N double, polarity-corrected morphology version
    mbeats_p  — 1×K integer, sample indices of detected R-peaks (at fs, 1-based)
    mbeats_q  — 1×K integer, sample indices of detected S-points (at fs, 1-based)
    R_amp     — scalar, median R-peak amplitude
    S_amp     — scalar, median S-point amplitude
    tfrrM     — freq×time double, cepstrum-weighted TFR (for HR curve)
    HR_ma     — 1×T double, estimated maternal heart rate (bins) per TF frame
    tfrtic    — freq×1 double, TFR frequency axis
    t         — 1×T double, TFR time axis

METHOD:
    Log-transform → CFPH (SST) → Extract_mhr (DP ridge) → beat_simple
    (DP beat tracker) → RRconstraint (refractory filter) → local peak search
"""

import numpy as np
import math
from scipy.signal import resample_poly
from scipy.interpolate import PchipInterpolator

from .CFPH import CFPH
from .extract_mhr import extract_mhr
from .beat_simple import beat_simple
from .RRconstraint import RRconstraint


def mbeats_extract_real(x0, x0_real, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat):
    """
    Extract maternal ECG beat locations from abdominal ECG.

    Parameters
    ----------
    x0 : ndarray
        Processed ECG signal (1-D), length N
    x0_real : ndarray
        Original ECG signal (1-D), length N
    fs : float
        Sampling frequency (Hz)
    basicTF : dict
        Basic STFT parameters (see CFPH)
    advTF : dict
        Advanced STFT parameters (see CFPH)
    cepR : dict
        Cepstrum parameters (see CFPH)
    P : dict
        Harmonic parameters (see CFPH)
    lam_curve : float
        Smoothness weight for CurveExt_M (HR curve extraction)
    lam_beat : float
        Transition-cost weight for beat_simple

    Returns
    -------
    x0 : ndarray
        Processed signal, possibly sign-flipped to canonical polarity
    x0_real : ndarray
        Original signal, possibly sign-flipped to canonical polarity
    mbeats_p : ndarray
        Sample indices of maternal R-peaks (1-based, matching MATLAB convention)
    mbeats_q : ndarray
        Sample indices of maternal S-points (1-based)
    R_amp : float
        Median R-peak amplitude
    S_amp : float
        Median S-point amplitude
    tfrrM : ndarray
        De-shaped maternal time-frequency representation
    HR_ma : ndarray
        Maternal heart rate curve (beats per frame)
    tfrtic : ndarray
        Frequency axis of the TFR (normalised)
    t : ndarray
        Time axis of the TFR (frame centres, in samples of 100 Hz)
    """
    # Ensure inputs are 1-D
    x0 = np.asarray(x0).flatten()
    x0_real = np.asarray(x0_real).flatten()

    # Truncate to a multiple of segment_size (200 ms at original fs) to keep
    # STFT frames aligned, then resample to 100 Hz for time-frequency analysis.
    segment_size = fs / 5  # segment length at original fs (200 ms)
    x0_len = int(np.floor(len(x0) / segment_size) * segment_size)
    x0 = x0[:x0_len]
    x0_real = x0_real[:x0_len]

    # Resample to 100 Hz using polyphase resampling (matches MATLAB resample)
    g = math.gcd(100, int(fs))
    x1 = resample_poly(x0, 100 // g, int(fs) // g)

    gg = np.log(1 + np.abs(x1))
    gg = gg - np.mean(gg)

    # Time-frequency analysis via CFPH
    _, _, _, tfrrM, _, _, tfrtic, t = CFPH(gg, basicTF, advTF, cepR, P)

    # Extract maternal HR curve from the de-shaped TFR
    HR_ma = extract_mhr(tfrrM, basicTF, lam_curve)

    # Interpolate HR curve to the original-fs domain (x0_len samples).

    segment_size_int = int(segment_size)
    x_hr_original = np.arange(segment_size_int, x0_len + 1, segment_size_int)

    # Ensure HR_ma has correct length (should match number of frame positions)
    if len(x_hr_original) > len(HR_ma):
        x_hr_original = x_hr_original[:len(HR_ma)]
    elif len(x_hr_original) < len(HR_ma):
        HR_ma = HR_ma[:len(x_hr_original)]

    # Interpolate in original-fs domain to all x0_len samples
    if len(x_hr_original) > 1 and len(HR_ma) > 1:
        f_hr = PchipInterpolator(x_hr_original, HR_ma, extrapolate=True)
        HR_ma2 = f_hr(np.arange(1, x0_len + 1))
    else:
        # Fallback if insufficient data for interpolation
        HR_ma2 = np.ones(x0_len) * (HR_ma[0] if len(HR_ma) > 0 else 1.0)

    # DP beat tracking (both polarities; choose the one with higher R-peak energy)
    mlocsf1p = beat_simple(x1, 100, HR_ma2 * basicTF['fr'], lam_beat)  # positive polarity
    mlocsf1q = beat_simple(-x1, 100, HR_ma2 * basicTF['fr'], lam_beat)  # negative polarity

    # Select polarity based on which detected peak set has higher median amplitude
    # Handle case where one polarity has no detections
    if len(mlocsf1p) == 0 and len(mlocsf1q) == 0:
        # No beats detected in either polarity
        Po = 1
        mlocsf1 = mlocsf1p
    elif len(mlocsf1p) == 0:
        # Only negative polarity detected beats
        Po = -1
        mlocsf1 = mlocsf1q
    elif len(mlocsf1q) == 0:
        # Only positive polarity detected beats
        Po = 1
        mlocsf1 = mlocsf1p
    else:
        # Both polarities detected beats - compare median amplitudes
        if np.abs(np.median(x1[mlocsf1p - 1])) > np.abs(np.median(x1[mlocsf1q - 1])):
            Po = 1
            mlocsf1 = mlocsf1p
        else:
            Po = -1
            mlocsf1 = mlocsf1q

    # Flip signals to canonical (upright R-peak) orientation
    x1 = Po * x1
    x0 = Po * x0
    x0_real = Po * x0_real

    # RR-interval constraint: remove the smaller-amplitude beat in any pair
    # separated by less than 250 ms (the physiological refractory period).
    mlocsf1 = RRconstraint(mlocsf1.astype(int), x1, 100, 0.25)

    # Scale beat locations from the 100 Hz resampled domain to the original sampling rate,
    # then refine to the nearest local max (R-peak) and min (S-point).
    x_up = x0
    scale_factor = fs / 100
    mlocsf = np.round(mlocsf1 * scale_factor).astype(int)

    # Adaptive search half-width for R-peak (scales with mean HR period);
    # fixed 48-sample half-width for S-point.
    SearchLen_p = int(np.round(48.0 / np.mean(HR_ma2 * basicTF['fr'])))
    SearchLen_q = 48

    # Refine beat locations by searching for local max (R-peak) and min (S-point)
    # near each DP-tracked beat location.
    # result = beat_idx - SearchLen + argmax(segment): when the window is clamped
    # to start_p=1, results < 1 are discarded by the `mbeats_p > 0` filter below.
    # Do not change start_p to 0-based or relax that filter — the formula relies on it.
    mbeats_p = []
    mbeats_q = []

    for ii in range(len(mlocsf)):
        beat_idx = mlocsf[ii]

        start_p = max(beat_idx - SearchLen_p, 1)
        end_p = min(beat_idx + SearchLen_p, len(x_up))
        segment_p = x_up[start_p - 1:end_p]
        idx_p = np.argmax(segment_p)
        mbeats_p_val = beat_idx - SearchLen_p + idx_p
        assert 1 <= mbeats_p_val <= len(x_up) or mbeats_p_val <= 0, (
            f"mbeats_extract_real: R-peak index {mbeats_p_val} out of expected range "
            f"for signal length {len(x_up)}; will be filtered by mbeats_p > 0 check"
        )
        mbeats_p.append(mbeats_p_val)

        start_q = max(beat_idx - SearchLen_q, 1)
        end_q = min(beat_idx + SearchLen_q, len(x_up))
        segment_q = x_up[start_q - 1:end_q]
        idx_q = np.argmin(segment_q)
        mbeats_q_val = beat_idx - SearchLen_q + idx_q
        mbeats_q.append(mbeats_q_val)

    mbeats_p = np.array(mbeats_p)
    mbeats_q = np.array(mbeats_q)

    # Discard out-of-bounds detections (match MATLAB: use < not <=)
    ind = (mbeats_p > 0) & (mbeats_p < x0_len) & (mbeats_q > 0) & (mbeats_q < x0_len)
    mbeats_p = mbeats_p[ind]
    mbeats_q = mbeats_q[ind]

    # Compute median beat amplitudes
    if len(mbeats_p) > 0:
        R_amp = np.median(x_up[mbeats_p - 1])
    else:
        R_amp = 0.0

    if len(mbeats_q) > 0:
        S_amp = np.median(x_up[mbeats_q - 1])
    else:
        S_amp = 0.0

    # Restore original polarity before returning
    x0 = Po * x0
    x0_real = Po * x0_real

    return x0, x0_real, mbeats_p, mbeats_q, R_amp, S_amp, tfrrM, HR_ma, tfrtic, t
