"""
RRconstraint: Remove spurious beat detections that violate the physiological refractory period.

For any pair of beats closer than RS·fs samples, discard the one with the smaller
absolute amplitude; uses setdiff-style removal (matching MATLAB's setdiff).
"""

import numpy as np


def RRconstraint(beats, xf, fs, RS):
    """
    Remove beats that violate a minimum RR-interval constraint.

    When two consecutive detected beats are separated by fewer than RS*fs
    samples (i.e. they are physiologically too close together), one of the
    two is discarded. The beat with the smaller absolute amplitude in the
    signal xf is removed, keeping the larger-amplitude candidate as the
    true R-peak.

    Parameters
    ----------
    beats : ndarray
        Beat sample indices (1-based, row or column vector, integer)
    xf : ndarray
        ECG signal vector; used to look up amplitude at each beat index
    fs : float
        Sampling frequency (Hz)
    RS : float
        Minimum RR interval expressed as a fraction of fs (seconds);
        pairs closer than RS*fs samples are considered duplicates

    Returns
    -------
    y : ndarray
        Pruned beat sample indices (same shape as input beats)
    """
    beats = np.asarray(beats, dtype=int).flatten()
    xf = np.asarray(xf).flatten()

    # Compute inter-beat intervals
    RRI = np.diff(beats)

    # Identify beats to remove in each too-close pair
    fp_list = []

    for ii in range(len(RRI)):
        if RRI[ii] <= RS * fs:
            # The pair (beats[ii], beats[ii+1]) is too close.
            # Keep the beat with the larger absolute amplitude; remove the other.
            if abs(xf[beats[ii] - 1]) <= abs(xf[beats[ii + 1] - 1]):
                fp_list.append(beats[ii])      # remove the earlier, smaller beat
            else:
                fp_list.append(beats[ii + 1])  # remove the later, smaller beat

    # Return beats with flagged duplicates removed
    fp = np.array(fp_list, dtype=int)
    y = np.setdiff1d(beats, fp)

    # Return as same shape as input (row or column)
    if beats.ndim == 1:
        return y
    else:
        return y.reshape(beats.shape)
