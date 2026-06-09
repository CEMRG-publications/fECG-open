"""
synchrosqueeze1win: Apply the Synchrosqueezing Transform (SST).

Reassigns STFT energy to the instantaneous frequency, sharpening spectral ridges.
A triangular smoothing window (Smooth=1, weight 0.25) is applied during accumulation.

Note: Smooth=1 is hardcoded (advTF.Smo=1 constant removed from main);
      Reject=0 block deleted (advTF.Rej=0 constant removed from main).

IMPORTANT: With Smooth=1, the triangular window triang(3)/sum(triang(3)) = [0.25;0.5;0.25]
is used. Only the edge weight (0.25) is applied at index new_idx. The center coefficient
(0.5) and second edge (0.25) are never accumulated because the loop runs zero iterations.
Net effect: the synchrosqueezed output is scaled by 0.25× rather than 1.0×.
This is an intentional consequence of the Smooth=1 design; do not "fix" the 0.25 factor.
"""

import numpy as np
from scipy.signal.windows import triang


def synchrosqueeze1win(tfr, ifd, alpha, fr, HighFreq, fs, ths):
    """
    Apply the Synchrosqueezing Transform (SST) to reassign STFT energy to the
    instantaneous frequency, sharpening spectral ridges.

    A triangular smoothing window (Smooth=1, weight 0.25) is applied during
    accumulation. Coefficients below the energy threshold ths are zeroed before
    reassignment.

    Parameters
    ----------
    tfr : ndarray
        STFT magnitude matrix (freq x time)
    ifd : ndarray
        Instantaneous frequency deviation (same size as tfr)
    alpha : float
        Frequency resolution (fr/fs)
    fr : float
        Frequency resolution in Hz (basicTF.fr)
    HighFreq : float
        Upper frequency limit as fraction of fs (advTF.HighFreq)
    fs : float
        Internal sampling rate (basicTF.fs = 100 Hz)
    ths : float
        Synchrosqueezing threshold: TFR entries below ths × mean(sum(|tfr|))
        are zeroed before reassignment (advTF.ths = 1e-6)

    Returns
    -------
    tfr : ndarray
        Input TFR after magnitude thresholding, trimmed to HighFreq
    rtfr : ndarray
        Synchrosqueezed TFR with triangular smoothing (0.25× centre weight)
    """
    # Smooth=1 is the fixed operating mode; half-width drives boundary guard and kernel
    Smooth = 1

    tfr = tfr.copy()
    ifd = ifd.copy()

    omega = ifd.copy()

    # Trim to HighFreq and round IFD for integer bin reassignment
    freq_idx_max = int(np.round(HighFreq * fs / fr))
    tfr = tfr[:freq_idx_max, :]
    omega = np.round(omega[:freq_idx_max, :]).astype(int)

    M, N = tfr.shape

    # Zero IFD values that would reassign a coefficient outside [1, M]
    OrigIndex = np.arange(1, M + 1)[:, np.newaxis] * np.ones((1, N))
    invalid_mask = (OrigIndex - omega < 1 + 2 * Smooth) | (OrigIndex - omega > M - 2 * Smooth)
    omega[invalid_mask] = 0

    # Magnitude thresholding
    Ex = np.mean(np.sum(np.abs(tfr), axis=0))
    Threshold = ths * Ex
    tfr[np.abs(tfr) < Threshold] = 0

    # Synchrosqueezing via accumarray (Fortran/column-major to match MATLAB)
    totLength = tfr.shape[0] * tfr.shape[1]
    tfr_flat = tfr.ravel(order='F')
    omega_flat = omega.ravel(order='F')

    new_idx = np.arange(1, totLength + 1) - omega_flat

    # Smoothed synchrosqueezing with triangular kernel (Smooth=1 → triang(3))
    SmoothWin = triang(1 + 2 * Smooth) / np.sum(triang(1 + 2 * Smooth))
    rtfr = np.zeros(totLength)

    # Center contribution
    indices_center = np.concatenate([[1], new_idx[1:-1], [totLength]]).astype(int)
    valid = (indices_center >= 1) & (indices_center <= totLength)
    np.add.at(rtfr, indices_center[valid] - 1, SmoothWin[0] * tfr_flat[valid])

    # With Smooth=1, range(1, 1) is empty — no off-center contributions needed

    rtfr = rtfr.reshape(tfr.shape[0], tfr.shape[1], order='F')

    return tfr, rtfr
