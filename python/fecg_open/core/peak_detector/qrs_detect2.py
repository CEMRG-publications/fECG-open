"""
qrs_detect2.py: Offline Pan-Tompkins QRS detector. 1:1 translation of qrs_detect2.m.
"""

import numpy as np
import math
from scipy.signal import filtfilt, resample_poly


# FIR bandpass filter coefficients designed at 250 Hz ("sombrero hat")
_B1_250HZ = np.array([
    -7.757327341237223e-05, -2.357742589814283e-04, -6.689305101192819e-04,
    -0.001770119249103, -0.004364327211358, -0.010013251577232, -0.021344241245400,
    -0.042182820580118, -0.077080889653194, -0.129740392318591, -0.200064921294891,
    -0.280328573340852, -0.352139052257134, -0.386867664739069, -0.351974030208595,
    -0.223363323458050, 0, 0.286427448595213, 0.574058766243311, 0.788100265785590,
    0.867325070584078, 0.788100265785590, 0.574058766243311, 0.286427448595213, 0,
    -0.223363323458050, -0.351974030208595, -0.386867664739069, -0.352139052257134,
    -0.280328573340852, -0.200064921294891, -0.129740392318591, -0.077080889653194,
    -0.042182820580118, -0.021344241245400, -0.010013251577232, -0.004364327211358,
    -0.001770119249103, -6.689305101192819e-04, -2.357742589814283e-04,
    -7.757327341237223e-05,
])


def _medfilt1_matlab(x, n):
    """
    1-D median filter matching MATLAB's medfilt1(x, n).

    MATLAB's medfilt1 zero-pads both ends of the signal before sliding the window.
    For odd n=2k+1, the window is [i-k : i+k] (k samples before, k after).
    For even n, the window is [i-n/2 : i+n/2-1] (n/2 before, n/2-1 after).
    In both cases the padding length is n//2 zeros on each side.
    """
    x = np.asarray(x, dtype=float).flatten()
    half = n // 2
    # Zero-pad (MATLAB behavior)
    padded = np.concatenate([np.zeros(half), x, np.zeros(half)])
    # Create strided view for vectorized median
    shape = (len(x), n)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.median(windows, axis=1)


def qrs_detect2(ecg, REF_PERIOD=0.250, THRES=0.6, fs=1000,
                fid_vec=None, SIGN_FORCE=None, debug=0, WIN_SAMP_SZ=7):
    """
    Pan-Tompkins QRS detector. 1:1 translation of qrs_detect2.m.

    Parameters
    ----------
    ecg : ndarray
        ECG signal (1-D)
    REF_PERIOD : float
        Refractory period (seconds, default 0.250)
    THRES : float
        Fraction of 98th percentile energy for threshold (default 0.6)
    fs : int
        Sampling frequency (Hz, default 1000)
    fid_vec : ndarray or None
        Indices to exclude from threshold estimation
    SIGN_FORCE : int or None
        Force peak polarity (+1/-1) or None for auto
    debug : int
        Unused (kept for interface compatibility)
    WIN_SAMP_SZ : int
        Integration window size in units of fs/256 (default 7)

    Returns
    -------
    qrs_pos : ndarray
        Sample indices of detected R-peaks
    sign : int
        Peak polarity (+1 or -1)
    en_thres : float
        Energy threshold used
    """
    ecg = np.asarray(ecg, dtype=float).flatten()
    NB_SAMP = len(ecg)

    tm = np.arange(1, NB_SAMP + 1) / fs

    # Algorithm constants
    MED_SMOOTH_NB_COEFF = int(round(fs / 100))
    INT_NB_COEFF = int(round(WIN_SAMP_SZ * fs / 256))
    SEARCH_BACK = True
    MIN_AMP = 0.1

    try:
        # Step 1: Bandpass filter — polyphase resampling of FIR coefficients to target fs
        g = math.gcd(int(fs), 250)
        b1 = resample_poly(_B1_250HZ, int(fs) // g, 250 // g)
        nf_b1 = max(len(b1), 1)
        bpfecg = filtfilt(b1, [1.0], ecg, padlen=3*(nf_b1-1))

        # Flatline guard (MATLAB line 127)
        if (np.sum(np.abs(ecg - np.median(ecg)) > MIN_AMP) / NB_SAMP) > 0.05:

            # Step 2: P&T operations
            dffecg = np.diff(bpfecg)
            sqrecg = dffecg * dffecg
            intecg = np.convolve(sqrecg, np.ones(INT_NB_COEFF), mode='full')[:len(sqrecg)]

            # Median smooth (MATLAB: medfilt1 zero-pads edges)
            mdfint = _medfilt1_matlab(intecg, MED_SMOOTH_NB_COEFF)

            # Compensate for integrator group delay
            delay = int(np.ceil(INT_NB_COEFF / 2))
            mdfint = np.roll(mdfint, -delay)

            # Fidelity vector
            if fid_vec is not None and len(fid_vec) > 0:
                mdfintFidel = mdfint.copy()
                fid_mask = fid_vec[fid_vec < len(mdfintFidel)]
                mdfintFidel[fid_mask[fid_mask > 2]] = 0
            else:
                mdfintFidel = mdfint

            # Step 3: Compute energy threshold
            if NB_SAMP / fs > 90:
                xs = np.sort(mdfintFidel[int(fs):int(fs * 90)])
            else:
                xs = np.sort(mdfintFidel[int(fs):])

            if len(xs) == 0:
                return np.array([], dtype=int), 1, 0.0

            if NB_SAMP / fs > 10:
                en_thres = xs[int(np.ceil(0.98 * len(xs))) - 1]
            else:
                en_thres = xs[int(np.ceil(0.99 * len(xs))) - 1]

            # Step 4: Threshold (MATLAB line 167)
            poss_reg = (mdfint > (THRES * en_thres)).astype(int)

            if np.sum(poss_reg) == 0:
                if len(poss_reg) > 10:
                    poss_reg[10] = 1

            # Step 5: Search-back
            if SEARCH_BACK:
                indAboveThreshold = np.where(poss_reg)[0]
                if len(indAboveThreshold) > 1:
                    # tm is 1-indexed in MATLAB; use tm[idx] for time
                    RRv = np.diff(tm[indAboveThreshold])
                    medRRv = np.median(RRv[RRv > 0.01]) if np.any(RRv > 0.01) else 1.0
                    indMissedBeat = np.where(RRv > 1.5 * medRRv)[0]

                    for i in indMissedBeat:
                        s = indAboveThreshold[i]
                        e = indAboveThreshold[i + 1]
                        poss_reg[s:e + 1] = (mdfint[s:e + 1] > (0.5 * THRES * en_thres)).astype(int)

            # Step 6: Find region boundaries
            left = np.where(np.diff(np.concatenate([[0], poss_reg])) == 1)[0]
            right = np.where(np.diff(np.concatenate([poss_reg, [0]])) == -1)[0]

            if len(left) == 0:
                return np.array([], dtype=int), 1, en_thres

            # Step 7: Determine peak polarity
            if SIGN_FORCE is not None and SIGN_FORCE != 0:
                sign = SIGN_FORCE
            else:
                nb_s = np.sum(left < 30 * fs)
                if nb_s == 0:
                    nb_s = len(left)
                loc = np.zeros(nb_s, dtype=int)
                for j in range(nb_s):
                    seg = np.abs(bpfecg[left[j]:right[j] + 1])
                    loc[j] = np.argmax(seg) + left[j]
                sign = np.mean(ecg[loc])

            # Step 8: Pick R-peak in each region
            maxval = []
            maxloc = []

            for i in range(len(left)):
                seg = ecg[left[i]:right[i] + 1]
                if sign > 0:
                    val = np.max(seg)
                    loc_rel = np.argmax(seg)
                else:
                    val = np.min(seg)
                    loc_rel = np.argmin(seg)
                loc_abs = loc_rel + left[i]

                # Refractory period check
                if len(maxloc) > 0:
                    tooClose = (loc_abs - maxloc[-1]) < fs * REF_PERIOD
                    if tooClose and abs(val) < abs(maxval[-1]):
                        continue  # discard current
                    elif tooClose and abs(val) >= abs(maxval[-1]):
                        maxloc[-1] = loc_abs
                        maxval[-1] = val
                    else:
                        maxloc.append(loc_abs)
                        maxval.append(val)
                else:
                    maxloc.append(loc_abs)
                    maxval.append(val)

            # Add 1 to return 1-based sample indices (following MATLAB convention)
            qrs_pos = np.array(maxloc, dtype=int) + 1
            return qrs_pos, sign, en_thres

        else:
            # Flatline
            return np.array([], dtype=int), 0, 0.0

    except Exception:
        return np.array([1, 10, 20], dtype=int), 1, 0.5
