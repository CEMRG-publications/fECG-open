"""
findQRSpeaks: Local maxima detection via sliding-window maximum.

A sample at position i is a peak if and only if it equals the maximum within
the window [i-win, i+win]. Returns 1-based indices (MATLAB convention).
"""

import numpy as np


def findQRSpeaks(ecg, win):
    """
    Find local maxima in ECG signal using sliding-window method.

    Parameters
    ----------
    ecg : ndarray
        ECG signal (1-D, any orientation)
    win : int
        Half-width of sliding maximum window (samples)

    Returns
    -------
    peakI : ndarray
        Sample indices of detected peaks (1-based, MATLAB convention, column vector)
    peakAmp : ndarray
        ECG amplitudes at detected peak indices (column vector)
    """

    ecg = np.asarray(ecg).flatten()
    L = len(ecg)

    # Pre-allocate boolean array to mark peaks
    is_peak = np.zeros(L, dtype=bool)

    # Sliding-window scan: a sample at position i is a peak if and only if it
    # equals the maximum within the window [i-win, i+win].
    for i in range(win, L - win):
        window_segment = ecg[i - win : i + win + 1]
        I = np.argmax(window_segment)
        if I == win:  # peak is at the centre of the window
            is_peak[i] = True

    peakI = np.where(is_peak)[0] + 1
    peakAmp = ecg[is_peak]  # amplitudes at peak positions

    return peakI.astype(int), peakAmp.astype(float)
