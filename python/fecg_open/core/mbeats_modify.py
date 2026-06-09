"""
mbeats_modify: Snap approximate R-peak locations to the nearest true local maximum or
minimum within a ±48-sample search window, then correct for polarity.

Method: Local max/min search in ±SearchLen window; swap R/S labels if polarity inverted.
"""

import numpy as np


def mbeats_modify(x_up, mlocsf):
    """
    Refine maternal beat locations to exact R-peak / S-point positions.

    Given a set of approximate beat locations (e.g. from a beat tracker),
    searches a ±48-sample window around each location to find:
      - mbeats_p : the local maximum (R-peak candidate)
      - mbeats_q : the local minimum (S-point / Q-trough candidate)

    After finding both, the function checks which set of points has a larger
    median absolute amplitude and assigns R vs S accordingly (correcting
    for polarity if necessary).

    Parameters
    ----------
    x_up : ndarray
        ECG signal vector (1D), at native sampling rate
    mlocsf : ndarray
        Approximate beat locations (1D integer vector, 1-based indices)

    Returns
    -------
    mbeats_p : ndarray
        Sample indices of R-peaks (local maxima), 1-based
    mbeats_q : ndarray
        Sample indices of S-points (local minima), 1-based
    R_amp : float
        Median amplitude at mbeats_p after polarity correction
    S_amp : float
        Median amplitude at mbeats_q after polarity correction
    Po : int
        Polarity flag: +1 if signal is upright, -1 if inverted
    """
    x0_len = len(x_up)
    x_up = np.asarray(x_up).flatten()
    mlocsf = np.asarray(mlocsf, dtype=int).flatten()

    mbeats_p = []
    mbeats_q = []
    SearchLen_p = 48   # search half-width for max (R-peak)
    SearchLen_q = 48   # search half-width for min (S-point)

    # Search for local max and min near each approximate beat location
    for ii in range(len(mlocsf)):
        a = max(mlocsf[ii] - SearchLen_p, 1)
        b = min(mlocsf[ii] + SearchLen_p, x0_len)
        segment = x_up[a - 1:b]
        idx_0based = np.argmax(segment)
        mbeats_p_val = mlocsf[ii] - SearchLen_p + idx_0based
        mbeats_p.append(mbeats_p_val)

        segment = x_up[a - 1:b]
        idx2_0based = np.argmin(segment)
        mbeats_q_val = mlocsf[ii] - SearchLen_q + idx2_0based
        mbeats_q.append(mbeats_q_val)

    mbeats_p = np.array(mbeats_p, dtype=int)
    mbeats_q = np.array(mbeats_q, dtype=int)

    # Discard detections outside valid signal range
    ind = np.where((mbeats_p > 0) & (mbeats_p < x0_len) & (mbeats_q > 0) & (mbeats_q < x0_len))[0]
    mbeats_p = mbeats_p[ind]
    mbeats_q = mbeats_q[ind]

    # Polarity correction
    # If the minimum set has larger median amplitude than the maximum set,
    # swap the labels (the signal is inverted relative to the expected R-peak)
    Po = 1
    if np.abs(np.median(x_up[mbeats_p - 1])) < np.abs(np.median(x_up[mbeats_q - 1])):
        mbeats_p, mbeats_q = mbeats_q.copy(), mbeats_p.copy()  # swap
        Po = -1
        x_up = -x_up

    # Compute median beat amplitudes after polarity correction
    R_amp = np.median(x_up[mbeats_p - 1])
    S_amp = np.median(x_up[mbeats_q - 1])

    return mbeats_p, mbeats_q, R_amp, S_amp, Po
