"""
pan_tompkin_revised: Full Pan-Tompkins QRS detector with adaptive thresholding.

Used as an independent cross-check for fetal beat detection alongside the SAVER
de-shape STFT pipeline. The polarity of the input signal must be corrected by
the caller before calling this function.

Method: findQRSpeaks → adaptive threshold (0.75·mean_8) → search-back at 1.5×mean_RR →
        inter-beat spacing guard (≥ 0.70·mean_RR).

NOTE: The Pan-Tompkins T-wave discrimination rule (accept candidate within 360 ms
only if its peak slope exceeds 50% of the prior QRS slope) is NOT implemented.
The variable 'skip' is reset to 0 in multiple places but is never set to 1 anywhere
in this function body (gr=1 plotting block removed: always called with gr=0 in this codebase).

Reference: Pan J, Tompkins WJ. A real-time QRS detection algorithm.
           IEEE Trans Biomed Eng. 32(3):230-236, 1985.
"""

import numpy as np
from .findQRSpeaks import findQRSpeaks


def pan_tompkin_revised(ecg, fs, gr=0):
    """
    Simplified Pan-Tompkins QRS detector with adaptive thresholding.

    Parameters
    ----------
    ecg : ndarray
        Raw ECG signal (1-D, any orientation)
    fs : float
        Sampling frequency (Hz)
    gr : int, optional
        If 1, show diagnostic plot (requires matplotlib); if 0, silent (default 0)

    Returns
    -------
    qrs_amp_raw : ndarray
        Amplitudes of detected R-waves
    qrs_i_raw : ndarray
        Sample indices of detected R-waves (1-based, MATLAB convention)
    delay : float
        Signal delay introduced by filtering (0 here)

    Notes
    -----
    Algorithm pipeline:
    1. Candidate peaks via findQRSpeaks (sliding-window maximum, win=0.2*fs).
    2. Initialise THR_SIG = max(first 2 s)/3 and THR_NOISE = mean(first 2 s)*3/4.
    3. Main loop over candidates: update running mean_RR and sigAmpThreshold
       (0.75 × mean of last 8 accepted amplitudes after 9 beats are detected).
    4. Search-back: if gap to current candidate > 1.5 × mean_RR, find the
       largest peak in the gap and insert it as an extra beat.
    5. Accept candidate if amplitude >= sigAmpThreshold AND spacing >= 0.7 × mean_RR.
    6. T-wave discrimination (360 ms slope rule) is NOT implemented; 'skip'
       is always 0 in this version.

    References
    ----------
    Pan J, Tompkins WJ. A real-time QRS detection algorithm.
    IEEE Trans Biomed Eng. 32(3):230-236, 1985.
    """

    # Input validation
    ecg = np.asarray(ecg).flatten()
    if ecg.ndim != 1:
        raise ValueError("ecg must be a 1-D vector")

    # Initialize state variables
    qrs_c = []  # amplitudes of accepted R-peaks
    qrs_i = []  # indices of accepted R-peaks (1-based)
    SIG_LEV = 0
    nois_c = []
    nois_i = []
    delay = 0
    skip = 0  # 1 when a T-wave is detected (skip next candidate)
    not_nois = 0  # 1 when search-back just added a beat
    selected_RR = []
    m_selected_RR = 0
    mean_RR = 0
    THRS_buf = []

    # Pre-select candidate peaks via sliding-window maximum
    locs, pks = findQRSpeaks(ecg, int(np.round(0.2 * fs)))

    # Training phase: initialise thresholds from first 2 seconds
    first_2s = int(2 * fs)
    first_2s = min(first_2s, len(ecg))  # guard against short signals

    THR_SIG = np.max(ecg[:first_2s]) * (1.0 / 3.0)
    THR_NOISE = np.mean(ecg[:first_2s]) * (3.0 / 4.0)
    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE
    sigAmpThreshold = THR_SIG
    mean_RR = 0

    # Main detection loop over candidate peaks
    for candidate_idx in range(len(pks)):
        y_i = pks[candidate_idx]  # amplitude
        x_i = locs[candidate_idx]  # index (1-based from findQRSpeaks)

        # Update running mean RR and amplitude threshold
        if len(qrs_c) >= 9:
            # Use last 9 beats for statistics
            diffRR = np.diff(np.array(qrs_i[-9:]))
            mean_RR = np.mean(diffRR)
            mean_8qrs = np.mean(np.array(qrs_c[-8:]))
            sigAmpThreshold = 0.75 * mean_8qrs
            comp = qrs_i[-1] - qrs_i[-2]
            m_selected_RR = mean_RR
        elif len(qrs_c) >= 1:
            # Use all beats so far
            if len(qrs_c) >= 2:
                diffRR = np.diff(np.array(qrs_i))
                mean_RR = np.mean(diffRR)
            else:
                mean_RR = 0
            mean_qrs = np.mean(np.array(qrs_c))
            sigAmpThreshold = 0.75 * mean_qrs

        # Choose which RR mean to use for search-back trigger
        if m_selected_RR:
            test_m = m_selected_RR
        elif mean_RR and m_selected_RR == 0:
            test_m = mean_RR
        else:
            test_m = 0

        # Search-back: recover missed beat if gap > 1.5 × mean_RR
        if test_m:
            if len(qrs_i) > 0 and (x_i - qrs_i[-1]) >= int(np.round(1.5 * test_m)):
                search_start = qrs_i[-1] + int(np.round(0.200 * fs))
                search_end = x_i - int(np.round(0.200 * fs))
                if search_start < search_end:
                    search_segment = ecg[search_start - 1 : search_end]
                    if len(search_segment) > 0:
                        pks_temp = np.max(search_segment)
                        locs_temp_0based = np.argmax(search_segment)
                        locs_temp = search_start + locs_temp_0based
                        qrs_c.append(pks_temp)
                        qrs_i.append(locs_temp)
                        not_nois = 1

        # Classify candidate: accept if amplitude >= sigAmpThreshold
        if y_i >= sigAmpThreshold:
            if len(qrs_i) > 1:
                # Reject if within 0.7 × mean_RR of the previous detection (T-wave guard)
                if (x_i - qrs_i[-1]) >= 0.70 * mean_RR and skip == 0:
                    qrs_c.append(y_i)
                    qrs_i.append(x_i)
            else:
                if skip == 0:
                    qrs_c.append(y_i)
                    qrs_i.append(x_i)

        # Reset per-iteration flags
        skip = 0
        not_nois = 0

        THRS_buf.append(sigAmpThreshold)

    qrs_i_raw = np.array(qrs_i, dtype=int)
    qrs_amp_raw = np.array(qrs_c, dtype=float)

    # Optional diagnostic plot (skipped in Python for simplicity when gr=0)
    if gr:
        try:
            import matplotlib.pyplot as plt

            bit2voltECG = ecg
            bit2voltThrs = np.array(THRS_buf)
            t = np.arange(1, len(ecg) + 1) / fs

            plt.figure()
            plt.plot(t, bit2voltECG, label="Filtered signal")
            plt.scatter(qrs_i_raw / fs, bit2voltECG[qrs_i_raw - 1], color="m", label="QRS peaks")
            plt.plot(locs / fs, bit2voltThrs, linewidth=2, linestyle="-.", color="g", label="Low threshold")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (mV)")
            plt.title("R peak detection with adaptive threshold")
            plt.legend()
            plt.grid(True)
            plt.show()
        except ImportError:
            print("matplotlib not available; skipping plot")

    return qrs_amp_raw, qrs_i_raw, delay
