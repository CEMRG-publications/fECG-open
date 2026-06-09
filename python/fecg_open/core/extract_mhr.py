"""
extract_mhr: Extract the maternal heart-rate curve from a synchrosqueezed TFR.

Note: MATLAB equivalent is named Extract_mhr (capitalised) in fecg_open/core/Extract_mhr.m (PEP 8 rename).

Masks the TFR to the physiological maternal HR band (0.4–2.4 Hz at basicTF.fs=100 Hz,
i.e. 24–144 bpm) and applies dynamic-programming ridge extraction (CurveExt_M).
"""

import numpy as np
from .CurveExt_M import CurveExt_M


def extract_mhr(rtfr_post, basicTF, dtw):
    """
    Extract maternal heart-rate curve from a synchrosqueezed TFR.

    Isolates the maternal heart-rate frequency band (0.4 – 2.4 Hz, i.e. 24 – 144 bpm)
    in the reassigned time-frequency representation, then extracts the dominant
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
        (1 x T) maternal heart-rate curve in TFR bin indices (1-based)
    """
    # Copy to avoid modifying input
    rtfr_post = rtfr_post.copy()

    # Suppress TFR energy outside the physiological maternal HR band (< 24 bpm and > 144 bpm).
    a = int(round(0.4 / basicTF['fr']))
    b = int(round(2.4 / basicTF['fr']))
    rtfr_post[0:a, :] = 0
    rtfr_post[b-1:, :] = 0

    # Extract the dominant maternal HR ridge using DP curve extraction.
    # Pass the first b rows (0 .. b-1 Python, 1 .. b MATLAB) to CurveExt_M.
    # Transpose because CurveExt_M expects (time x frequency).
    HR = CurveExt_M(rtfr_post[0:b, :].T, dtw)[0]

    return HR
