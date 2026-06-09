"""
fbeats_extract_real: Extract fetal ECG R-peaks from a rough fECG residual signal,
suppressing any residual maternal HR energy in the time-frequency plane before tracking.

INPUTS:
    I2_orig       — 1×N double, rough fECG residual (for timing)
    I2_orig_morph — 1×N double, morphology-preserving fECG residual
    fs            — scalar, sampling rate (Hz), nominally 1000
    basicTF       — dict: STFT parameters (win compressed to 3/5 for fetal analysis)
    advTF         — dict: SST thresholds
    cepR          — dict: cepstral parameters
    P             — dict: P['num_s']=1
    lam_curve     — scalar >=0, DP smoothness weight for HR ridge
    lam_beat      — scalar >=0, DP transition weight for beat tracker
    HR_ma         — 1×T double, maternal HR (bins) per TF frame — used to suppress
                    maternal residue in the fetal TFR (±6% around each maternal bin)

OUTPUTS: (same layout as mbeats_extract_real but for fetal beats)
    I2_orig       — polarity-corrected fECG residual
    I2_orig_morph — polarity-corrected morphology residual
    fbeats_p      — 1×K integer, fetal R-peak sample indices (1-based)
    fbeats_q      — 1×K integer, fetal S-point sample indices (1-based)
    R_amp / S_amp — scalar amplitudes
    tfrrF         — freq×time, fetal TFR after maternal suppression
    HR_fe         — 1×T double, estimated fetal HR (bins) per frame
    tfrtic / t    — frequency and time axes

METHOD:
    Resample to 100 Hz → CFPH (SST, window = 3/5 × basicTF.win) →
    suppress maternal HR band (±6%) in TFR → Extract_fhr (DP ridge) →
    beat_simple → RRconstraint → local peak search at native sampling rate

Note: The STFT window is narrowed to 3/5 of basicTF.win for fetal analysis
      (shorter window → better frequency resolution at higher fetal HR).
"""

import numpy as np
import math
from scipy.signal import resample_poly
from scipy.interpolate import PchipInterpolator, interp1d

from .CFPH import CFPH
from .extract_fhr import extract_fhr
from .beat_simple import beat_simple
from .RRconstraint import RRconstraint


def fbeats_extract_real(I2_orig, I2_orig_morph, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat, HR_ma):
    """
    Extract fetal ECG beat locations from residual ECG signal.

    Parameters
    ----------
    I2_orig : ndarray
        Fetal ECG residual signal (1-D), length N, native sample rate fs
    I2_orig_morph : ndarray
        Morphology-preserving version of the fetal ECG (1-D), length N
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
        Smoothness weight for CurveExt_M
    lam_beat : float
        Transition-cost weight for beat_simple
    HR_ma : ndarray
        Maternal HR curve (beats per frame, length = TFR columns)
        Used to suppress the maternal frequency band in the TFR

    Returns
    -------
    I2_orig : ndarray
        Fetal signal, possibly sign-flipped
    I2_orig_morph : ndarray
        Morphology-preserving signal, possibly sign-flipped
    fbeats_p : ndarray
        Sample indices of fetal R-peaks (1-based)
    fbeats_q : ndarray
        Sample indices of fetal S-points (1-based)
    R_amp : float
        Median fetal R-peak amplitude
    S_amp : float
        Median fetal S-point amplitude
    tfrrF : ndarray
        De-shaped fetal time-frequency representation
    HR_fe : ndarray
        Fetal heart rate curve (beats per frame)
    tfrtic : ndarray
        Frequency axis of the TFR
    t : ndarray
        Time axis of the TFR (frame centres)
    """
    # Ensure inputs are 1-D
    I2_orig = np.asarray(I2_orig).flatten()
    I2_orig_morph = np.asarray(I2_orig_morph).flatten()
    HR_ma = np.asarray(HR_ma).flatten()

    # Resample to 100 Hz for TF analysis using polyphase resampling (matches MATLAB resample)
    I2_orig_len = int(np.floor(len(I2_orig) / 200) * 200)
    g = math.gcd(100, int(fs))
    I2 = resample_poly(I2_orig, 100 // g, int(fs) // g)

    # Narrow the STFT window for the fetal signal (higher frequency resolution)
    basicTF = basicTF.copy()
    basicTF["win"] = int(np.round(basicTF["win"] * 3 / 5))

    gg = I2.copy()
    gg = gg - np.mean(gg)

    # Time-frequency analysis
    _, _, _, tfrrF, _, _, tfrtic, t = CFPH(gg, basicTF, advTF, cepR, P)

    # Attenuate TFR bins within ±6% of the maternal HR frequency bin per time frame.
    # HR_ma is in TFR bin units (1-based, matching MATLAB convention).
    for ti in range(tfrrF.shape[1]):
        idx_ma_hz = HR_ma[min(ti, len(HR_ma) - 1)]
        # MATLAB: round(HR_ma(ti)*0.94):round(HR_ma(ti)*1.06) produces 1-based frequency bin
        #         indices used directly in 1-based MATLAB array indexing
        # Python: -1 corrects to 0-based indexing
        idx_lower = int(np.round(idx_ma_hz * 0.94)) - 1
        idx_upper = int(np.round(idx_ma_hz * 1.06)) - 1
        if idx_upper >= tfrrF.shape[0] or idx_lower < 0:
            continue  # Skip if maternal HR exceeds TFR range

        # Attenuate maternal band
        tfrrF[idx_lower:idx_upper + 1, ti] = (tfrrF[idx_lower:idx_upper + 1, ti] / 10.0)

    # Extract fetal HR curve
    HR_fe = extract_fhr(tfrrF, basicTF, lam_curve)

    # Interpolate fetal HR to the original-fs domain (I2_orig_len samples).
    segment_size_fe = 200
    x_hr_fe = np.arange(segment_size_fe, len(I2_orig) + 1, segment_size_fe)
    if len(x_hr_fe) > len(HR_fe):
        x_hr_fe = x_hr_fe[: len(HR_fe)]
    elif len(HR_fe) > len(x_hr_fe):
        HR_fe = HR_fe[: len(x_hr_fe)]
    if len(x_hr_fe) > 1 and len(HR_fe) > 1:
        f_hr_fe = PchipInterpolator(x_hr_fe, HR_fe, extrapolate=True)
        HR_fe3 = f_hr_fe(np.arange(1, len(I2_orig) + 1))
    else:
        HR_fe3 = np.ones(len(I2_orig)) * (HR_fe[0] if len(HR_fe) > 0 else 2.0)

    # DP beat tracking (both polarities)
    flocsf1p = beat_simple(I2, 100, HR_fe3 * basicTF["fr"], lam_beat)
    flocsf1q = beat_simple(-I2, 100, HR_fe3 * basicTF["fr"], lam_beat)

    # Polarity selection
    if len(flocsf1p) > 0 and len(flocsf1q) > 0:
        median_p = np.median(I2[flocsf1p - 1])
        median_q = np.median(I2[flocsf1q - 1])
        if np.abs(median_p) > np.abs(median_q):
            Po = 1
            flocsf = flocsf1p
        else:
            Po = -1
            flocsf = flocsf1q
            print("\t\t*** reverse the fetal pole")
    elif len(flocsf1p) > 0:
        Po = 1
        flocsf = flocsf1p
    else:
        Po = -1
        flocsf = flocsf1q

    # Apply polarity flip
    I2 = Po * I2
    I2_orig = Po * I2_orig
    I2_orig_morph = Po * I2_orig_morph

    # RR-interval constraint
    flocsf = RRconstraint(flocsf.astype(int), I2, 100, 0.25)

    # Refine beat locations
    I2 = I2_orig
    flocsf = np.round(flocsf * 10).astype(int)  # scale from 100 Hz to native fs

    fbeats_p = []
    fbeats_q = []
    SearchLen_p = int(np.round(100.0 / np.mean(HR_fe3 * basicTF["fr"])))  # adaptive R-peak window
    SearchLen_q = 100  # fixed S-point window

    for ii in range(len(flocsf)):
        # Ensure beat position is within bounds
        beat_pos = flocsf[ii]
        if beat_pos < 1 or beat_pos > len(I2):
            continue  # Skip out-of-bounds beats

        # R-peak: local maximum
        start_p = max(beat_pos - SearchLen_p, 1)
        end_p = min(beat_pos + SearchLen_p, len(I2))

        if start_p > end_p or start_p > len(I2):
            continue

        segment = I2[start_p - 1:end_p]
        if len(segment) == 0:
            continue
        idx_p = np.argmax(segment)
        fbeats_p_val = beat_pos - SearchLen_p + idx_p
        fbeats_p.append(fbeats_p_val)

        # S-point: local minimum
        start_q = max(beat_pos - SearchLen_q, 1)
        end_q = min(beat_pos + SearchLen_q, len(I2))

        if start_q > end_q or start_q > len(I2):
            continue

        segment_q = I2[start_q - 1:end_q]
        if len(segment_q) == 0:
            continue
        idx_q = np.argmin(segment_q)
        fbeats_q_val = beat_pos - SearchLen_q + idx_q
        fbeats_q.append(fbeats_q_val)

    fbeats_p = np.array(fbeats_p)
    fbeats_q = np.array(fbeats_q)

    # Discard out-of-bounds detections
    ind = ((fbeats_p > 0) & (fbeats_p < I2_orig_len) & (fbeats_q > 0) & (fbeats_q < I2_orig_len))
    fbeats_p = fbeats_p[ind]
    fbeats_q = fbeats_q[ind]

    # Compute median amplitudes
    if len(fbeats_p) > 0:
        R_amp = np.median(I2[fbeats_p - 1])
    else:
        R_amp = 0.0

    if len(fbeats_q) > 0:
        S_amp = np.median(I2[fbeats_q - 1])
    else:
        S_amp = 0.0

    # Restore original polarity
    I2_orig = Po * I2_orig
    I2_orig_morph = Po * I2_orig_morph

    return I2_orig, I2_orig_morph, fbeats_p, fbeats_q, R_amp, S_amp, tfrrF, HR_fe, tfrtic, t
