"""
extract_fhr: Extract the fetal heart-rate curve from a synchrosqueezed TFR.

Note: MATLAB equivalent is named Extract_fhr (capitalised) in fecg_open/core/Extract_fhr.m (PEP 8 rename).

Two-stage frequency masking with different bounds:
  1) Masking (0.4-3.2 Hz): rows of the full TFR outside this band are zeroed
     to suppress DC drift/baseline wander (<0.4 Hz) and high-frequency noise
     (>3.2 Hz). The retained band still contains maternal HR energy
     (maternal band: 0.4-2.4 Hz).
  2) DP ridge search (1.0-3.2 Hz): a 1.0-3.2 Hz submatrix of the masked TFR
     is passed to CurveExt_M for dynamic-programming ridge tracking. The
     lower bound is raised to 1.0 Hz (from the 0.4 Hz masking bound) to
     exclude the maternal fundamental and its lower harmonics, which would
     otherwise contaminate the fetal ridge estimate. Returned bin indices
     are corrected back to full-TFR coordinates by adding round(1/basicTF.fr).
"""

import numpy as np
from .CurveExt_M import CurveExt_M


def extract_fhr(rtfr_post, basicTF, dtw):
    """
    Extract fetal heart-rate curve from a synchrosqueezed TFR.

    Stage 1 masks the TFR to 0.4-3.2 Hz (suppresses baseline wander and
    high-frequency noise; the retained band still contains maternal HR
    energy, 0.4-2.4 Hz). Stage 2 extracts the 1.0-3.2 Hz submatrix from the
    masked TFR and passes it to CurveExt_M for dynamic-programming ridge
    tracking; the 1.0 Hz lower bound excludes the maternal fundamental and
    its lower harmonics. Returned bin indices are offset back to full-TFR
    coordinates by adding round(1/basicTF.fr).

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
