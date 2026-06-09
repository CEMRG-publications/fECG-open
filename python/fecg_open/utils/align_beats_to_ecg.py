"""
align_beats_to_ecg: Fine-tune beat sample indices by snapping each to the nearest local
maximum (upright signal) or minimum (inverted signal) within a small search window.

Corrects ±few-sample timing errors introduced by the beat tracker.
Polarity is determined from mean(ecg[beats]): positive → find local max; negative → find local min.

Callers: mbeats use sample_window=5; fbeats use sample_window=2.
"""

import numpy as np


def align_beats_to_ecg(beats, ecg, sample_window):
    """
    Snap beat indices to nearest local extremum in ECG signal.

    Parameters
    ----------
    beats : ndarray
        Beat sample indices (1-based, MATLAB convention)
    ecg : ndarray
        ECG signal vector (1-D)
    sample_window : int
        Search half-width in samples

    Returns
    -------
    beats_new : ndarray
        Updated beat sample indices (1-based)
    changed_ctr : int
        Number of beats whose location was moved
    """

    beats = np.asarray(beats).flatten().astype(int)
    ecg = np.asarray(ecg).flatten()

    beats_new = beats.copy()
    # Determine signal polarity from the mean ECG value at all beat positions.
    # Positive → R-peaks are upright; negative → R-peaks are inverted troughs.
    Po = np.mean(ecg[beats - 1]) > 0
    changed_ctr = 0

    for i in range(len(beats_new)):
        beat_idx = beats_new[i]

        # Clamp search window to signal boundaries.
        window_lower = max(beat_idx - sample_window - 1, 0)
        window_upper = min(beat_idx + sample_window, len(ecg))

        if Po > 0:
            # Upright signal: snap to local maximum
            window_segment = ecg[window_lower:window_upper]
            max_idx_local = np.argmax(window_segment)
            max_idx = max_idx_local + window_lower + 1

            if ecg[max_idx - 1] > ecg[beat_idx - 1]:
                beats_new[i] = max_idx
                changed_ctr += 1
        else:
            # Inverted signal: snap to local minimum
            window_segment = ecg[window_lower:window_upper]
            min_idx_local = np.argmin(window_segment)
            min_idx = min_idx_local + window_lower + 1

            if ecg[min_idx - 1] < ecg[beat_idx - 1]:
                beats_new[i] = min_idx
                changed_ctr += 1

    return beats_new, changed_ctr
