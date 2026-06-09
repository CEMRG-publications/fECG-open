"""
extract_fhr: Extract the fetal heart-rate curve from a synchrosqueezed TFR.

Note: MATLAB equivalent is named Extract_fhr (capitalised) in fecg_open/core/Extract_fhr.m (PEP 8 rename).

Masks the TFR to the fetal HR band (1.0–3.2 Hz at basicTF.fs=100 Hz, i.e. 60–192 bpm)
and applies dynamic-programming ridge extraction (CurveExt_M). The 1.0 Hz lower cutoff
targets the typical fetal HR range (~1.5–2.7 Hz, 90–160 BPM), which lies above the
typical maternal HR range (0.67–1.67 Hz, 40–100 BPM).

These ranges can overlap in pathological conditions (e.g. fetal bradycardia <80 BPM,
maternal tachycardia >100 BPM); the cutoffs are heuristics tuned for typical physiology.
"""

import numpy as np
from .CurveExt_M import CurveExt_M


def extract_fhr(rtfr_post, basicTF, dtw):
    """
    Extract fetal heart-rate curve from a synchrosqueezed TFR.

    Isolates the fetal heart-rate frequency band (0.4 – 3.2 Hz) in the
    reassigned time-frequency representation, then extracts the dominant
    instantaneous frequency trajectory using the dynamic-programming curve
    extractor CurveExt_M.

    The extracted curve HR is expressed in TFR frequency-bin units.
    Convert to BPM with:  bpm = basicTF['fs'] / HR / basicTF['fr'] * 60

    Parameters
    ----------
    rtfr_post : ndarray
        (F x T) reassigned TFR matrix (frequency bins x time frames)
    basicTF : dict
        Dictionary with fields:
          'fr'  – frequency resolution (Hz / bin)
          'fs'  – sampling frequency of the original signal (Hz)
    dtw : float
        Dynamic-programming cost weight (see CurveExt_M)

    Returns
    -------
    HR : ndarray
        (1 x T) fetal heart-rate curve in TFR bin indices (1-based)
    """
    # Copy to avoid modifying input
    rtfr_post = rtfr_post.copy()

    # Suppress TFR energy outside the physiological fetal HR band (< 24 bpm and > 192 bpm).
    a = int(round(0.4 / basicTF['fr']))
    b = int(round(3.2 / basicTF['fr']))
    rtfr_post[0:a, :] = 0
    rtfr_post[b-1:, :] = 0

    # Feed only the 1–3.2 Hz sub-matrix into CurveExt_M to prevent residual
    # maternal HR energy (< 1 Hz) from confusing the DP tracker.
    start_idx = int(round(1.0 / basicTF['fr']))
    end_idx = int(round(3.2 / basicTF['fr']))
    HR = CurveExt_M(rtfr_post[start_idx:end_idx, :].T, dtw)[0]

    # Offset HR so that returned indices refer to the full TFR frequency axis,
    # not just the sub-matrix passed to CurveExt_M.
    HR = HR + start_idx

    return HR
