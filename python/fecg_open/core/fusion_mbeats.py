"""
fusion_mbeats: Fuse R-peak lists from multiple signal orientations into a single
consensus beat sequence.

Three-stage algorithm (Reference: Qioung Yu, 2016):
  Stage 1 — filter orientations with implausible RRI (0.4–1.6 s); score remaining
             by template correlation and RRI regularity.
  Stage 2 — keep orientations scoring ≥ 50% of best score; refine beat sets by
             per-beat correlation threshold (th3_3=0.5).
  Stage 3 — cluster peaks within 30 ms; keep clusters present in ≥ nb_c/2
             orientations; report cluster median as final beat.
"""

import numpy as np
from scipy.stats import pearsonr


def fusion_mbeats(mbeats_c, aECG_c, fs):
    """
    Multi-channel fusion of maternal R-peak detections.

    Parameters
    ----------
    mbeats_c : list of ndarray
        Cell array of beat indices for each channel (1 x nb_c), each with 1-based indices
    aECG_c : list of ndarray
        Cell array of abdominal ECG signals for each channel (1 x nb_c)
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    mbeats : ndarray
        Fused R-peak sample indices (1-based, sorted)
    """

    # Thresholds and parameters (matching MATLAB exactly)
    th2_1 = 0.6         # minimum template correlation for a "good" beat (Stage 1)
    th2_2 = 0.06 * fs   # RR-median deviation threshold for score2 (Stage 1)
    th3_3 = 0.5         # per-beat template correlation threshold (Stage 2)
    a = 2.5             # RR-variability multiplier for Stage 3 clustering window

    nb_c = len(mbeats_c)

    # ========================================================================
    # Stage 1a: discard channels with pathological RR intervals
    # ========================================================================
    ind = []            # surviving channel indices
    RRI_med_c = []      # median RRI per surviving channel

    # Try with standard bounds first
    for i in range(nb_c):
        if len(mbeats_c[i]) > 0:
            RRI = np.median(np.diff(mbeats_c[i]))
            # Keep channel only if median RRI is within physiological bounds
            if RRI < 0.4*fs or RRI > 1.6*fs:
                continue
            ind.append(i)
            RRI_med_c.append(RRI)

    if len(ind) == 0:
        return np.array([], dtype=int)

    nb_c = len(ind)
    RRI_med_c = np.array(RRI_med_c)
    RRI_m = np.median(RRI_med_c)  # overall median RRI
    MaximalQT = int(np.ceil(np.median(RRI_m) / 2))

    # Deviation of each channel's median RRI from the overall median
    med_diff_c = np.abs(RRI_med_c - RRI_m)
    mi = np.min(med_diff_c)
    error2_c = np.abs(med_diff_c - mi)

    # ========================================================================
    # Stage 1b: score each channel and discard low-scorers
    # ========================================================================
    score_c = []

    for i in range(nb_c):
        sub_i = ind[i]
        mbeats = mbeats_c[sub_i].copy()
        I = aECG_c[sub_i].copy()
        RRI = np.diff(mbeats)
        RRI = np.concatenate(([RRI[0]], RRI))  # prepend first interval

        # Trim boundary beats
        tmp = np.where(mbeats > MaximalQT)[0]
        mbeats = mbeats[tmp]
        RRI = RRI[tmp]

        tmp = np.where(mbeats + MaximalQT <= len(I))[0]
        mbeats = mbeats[tmp]
        RRI = RRI[tmp]

        # Build beat matrix and compute correlation with median template
        V = []
        for ii in range(len(RRI)):
            idx_start = int(mbeats[ii] - MaximalQT - 1)
            idx_end = int(mbeats[ii] + MaximalQT)
            if idx_start >= 0 and idx_end <= len(I):
                V.append(I[idx_start:idx_end])

        if len(V) == 0:
            score_c.append(0)
            continue

        V = np.array(V).T  # shape: (window_size, num_beats)
        template = np.median(V, axis=1)

        # Correlation of each beat with template
        score1_list = []
        for j in range(V.shape[1]):
            try:
                corr = np.corrcoef(template, V[:, j])[0, 1]
                if not np.isnan(corr):
                    score1_list.append(corr)
                else:
                    score1_list.append(0)
            except:
                score1_list.append(0)

        score1_list = np.array(score1_list)
        score1 = np.sum(score1_list >= th2_1) / len(score1_list) if len(score1_list) > 0 else 0

        # RRI proximity score (discrete: 0.3, 0.2, or 0.1)
        if error2_c[i] < th2_2:
            score2 = 0.3
        elif error2_c[i] < 2*th2_2:
            score2 = 0.2
        else:
            score2 = 0.1

        score_c.append(score1 * score2)

    score_c = np.array(score_c)

    # Keep channels with score >= 50% of the best score
    if len(score_c) == 0 or np.max(score_c) == 0:
        return np.array([], dtype=int)

    ma = np.max(score_c)
    mask = score_c >= 0.5*ma
    ind_new = [ind[i] for i in range(len(ind)) if mask[i]]
    RRI_med_c = RRI_med_c[mask]
    RRI_m2 = np.median(RRI_med_c)
    nb_c = len(ind_new)
    MaximalQT = int(np.ceil(np.median(RRI_m2) / 2))
    ind = ind_new

    if nb_c == 0:
        return np.array([], dtype=int)

    # ========================================================================
    # Stage 2: per-beat correlation filtering within surviving channels
    # ========================================================================
    mbeats_c_filtered = {}

    for i in range(nb_c):
        sub_i = ind[i]
        mbeats = mbeats_c[sub_i].copy()
        I = aECG_c[sub_i].copy()
        RRI = np.diff(mbeats)
        RRI = np.concatenate(([RRI[0]], RRI))

        tmp = np.where(mbeats > MaximalQT)[0]
        mbeats = mbeats[tmp]
        RRI = RRI[tmp]

        tmp = np.where(mbeats + MaximalQT <= len(I))[0]
        mbeats = mbeats[tmp]
        RRI = RRI[tmp]

        # Build beat matrix
        V = []
        for ii in range(len(RRI)):
            idx_start = int(mbeats[ii] - MaximalQT - 1)
            idx_end = int(mbeats[ii] + MaximalQT)
            if idx_start >= 0 and idx_end <= len(I):
                V.append(I[idx_start:idx_end])

        if len(V) == 0:
            mbeats_c_filtered[sub_i] = np.array([], dtype=int)
            continue

        V = np.array(V).T
        template = np.median(V, axis=1)

        # Correlation with template
        score3_list = []
        for j in range(V.shape[1]):
            try:
                corr = np.corrcoef(template, V[:, j])[0, 1]
                if not np.isnan(corr):
                    score3_list.append(corr)
                else:
                    score3_list.append(0)
            except:
                score3_list.append(0)

        score3 = np.array(score3_list)

        # Keep only beats with correlation > th3_3
        keep_beats = mbeats[score3 > th3_3]
        mbeats_c_filtered[sub_i] = keep_beats

   
    # ========================================================================
    # Stage 3: Pool all beats and cluster by proximity
    # ========================================================================
    X = []
    for i in range(nb_c):
        sub_i = ind[i]
        X.extend(mbeats_c_filtered[sub_i].tolist())

    if len(X) == 0:
        return np.array([], dtype=int)

    R_peak = np.sort(X)

    # Group beats that fall within 30 ms (0.03*fs samples) of each other
    Q = []
    Q.append([R_peak[0]])
    for j in range(1, len(R_peak)):
        if abs(R_peak[j] - R_peak[j-1]) <= 0.03*fs:
            Q[-1].append(R_peak[j])
        else:
            Q.append([R_peak[j]])

    # Accept a cluster only if it was seen in >= half the surviving channels
    mbeats = []
    for cluster in Q:
        if len(cluster) >= nb_c / 2:
            # Use MATLAB-style rounding (away from zero) instead of banker's rounding
            # For positive values: int(floor(x + 0.5)) rounds away from zero
            median_val = np.median(cluster)
            mbeats.append(int(np.floor(median_val + 0.5)))

    return np.array(mbeats, dtype=int)
