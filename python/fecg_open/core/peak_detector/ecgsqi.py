"""
ecgsqi.py: F1-based windowed beat-agreement SQI. 1:1 translation of ecgsqi.m.
"""

import numpy as np

def _histc(data, edges):
    """
    Emulate MATLAB's histc(data, edges).

    histc counts elements of data falling in bins:
      bin k = [edges(k), edges(k+1))  for k = 1..n-1
      bin n = elements == edges(n)     (last bin is point-count)

    Returns array of length len(edges).
    """
    data = np.asarray(data, dtype=float).flatten()
    edges = np.asarray(edges, dtype=float).flatten()
    counts = np.zeros(len(edges), dtype=int)
    for k in range(len(edges) - 1):
        counts[k] = np.sum((data >= edges[k]) & (data < edges[k + 1]))
    # Last bin: exact match
    counts[-1] = np.sum(data == edges[-1])
    return counts


def ecgsqi(ann1, ann2, THR, SIZE_WIND, REG_WIN, LG_MED, LG_REC, N_WIN):
    """
    Windowed beat-agreement SQI. 1:1 translation of ecgsqi.m.

    Parameters
    ----------
    ann1 : ndarray
        Beat times (seconds) from detector 1
    ann2 : ndarray
        Beat times (seconds) from detector 2
    THR : float
        Acceptance half-window (seconds)
    SIZE_WIND : float
        SQI window length (seconds)
    REG_WIN : float
        SQI window stride (seconds)
    LG_MED : int
        Half-width of sliding minimum smoothing
    LG_REC : float
        Recording length (seconds)
    N_WIN : int
        Expected number of SQI windows

    Returns
    -------
    sqi : ndarray
        Smoothed F1 SQI values
    tsqi : ndarray
        Window start times
    """
    ann1 = np.asarray(ann1, dtype=float).flatten()
    ann2 = np.asarray(ann2, dtype=float).flatten()

    # Step 1: Count ann2 beats in windows around ann1
    if len(ann1) > 0:
        xi = np.column_stack([ann1 - THR, ann1 + THR]).flatten()

        # Fix overlapping windows — BATCH (matches MATLAB simultaneous assignment).
        _diff = np.diff(xi)
        _idx_fix  = np.concatenate([[False], _diff < 0])   # True where xi[k] < xi[k-1]
        _idx_prev = np.concatenate([_idx_fix[1:], [False]]) # True at each predecessor
        if _idx_fix.any():
            _xi_avg = (xi[_idx_fix] + xi[_idx_prev]) / 2.0
            xi = xi.copy()
            xi[_idx_fix]  = _xi_avg   # xi(idxFix)                    = xi_fixed
            xi[_idx_prev] = _xi_avg   # xi([idxFix(2:end); false])    = xi_fixed

        N_J = _histc(ann2, xi)
        N_J = N_J[0::2]  # Keep odd-indexed bins (MATLAB line 69)
    else:
        N_J = np.array([])

    # Step 2: Count ann1 beats in windows around ann2
    if len(ann2) > 0:
        xi2 = np.column_stack([ann2 - THR, ann2 + THR]).flatten()

        _diff2 = np.diff(xi2)
        _idx_fix2  = np.concatenate([[False], _diff2 < 0])
        _idx_prev2 = np.concatenate([_idx_fix2[1:], [False]])
        if _idx_fix2.any():
            _xi2_avg = (xi2[_idx_fix2] + xi2[_idx_prev2]) / 2.0
            xi2 = xi2.copy()
            xi2[_idx_fix2]  = _xi2_avg
            xi2[_idx_prev2] = _xi2_avg

        N_G = _histc(ann1, xi2)
        N_G = N_G[0::2]
    else:
        N_G = np.array([])

    # Step 3: Aggregate into SQI windows
    xi1 = np.arange(0, LG_REC + REG_WIN, REG_WIN)
    xi2_win = xi1 + SIZE_WIND

    # Handle edge effect
    xi1_trunc = xi1.copy()
    xi1_trunc[xi2_win > LG_REC] = LG_REC - SIZE_WIND

    F1_1 = np.zeros(N_WIN)
    F1_2 = np.zeros(N_WIN)

    for w in range(min(len(xi1_trunc), N_WIN)):
        idx1 = (ann1 > xi1_trunc[w]) & (ann1 < xi2_win[w])
        idx2 = (ann2 > xi1_trunc[w]) & (ann2 < xi2_win[w])

        nj_in = N_J[idx1] if len(N_J) > 0 else np.array([])
        ng_in = N_G[idx2] if len(N_G) > 0 else np.array([])

        # MATLAB mean([]) = NaN; return NaN for empty windows so np.fmin below
        # can propagate the non-NaN value (matching MATLAB min(x, NaN) = x behaviour).
        F1_1[w] = np.mean(nj_in == 1) if len(nj_in) > 0 else float('nan')
        F1_2[w] = np.mean(ng_in == 1) if len(ng_in) > 0 else float('nan')

    # MATLAB: min(F1_1, NaN) = F1_1 (NaN-ignoring); MATLAB mean([]) returns NaN
    # Python: np.fmin matches MATLAB's NaN-ignoring semantics: fmin(x, NaN) = x
    F1 = np.fmin(F1_1, F1_2)
    F1[np.isnan(F1)] = 0

    # Remove windows beyond record end
    valid = xi1[:N_WIN] < LG_REC
    F1 = F1[valid]
    xi1_out = xi1[:N_WIN][valid]

    # Step 4: Sliding minimum smoothing
    if len(F1) < (LG_MED * 2 + 1):
        F1smooth = F1
    else:
        cols = 2 * LG_MED + 1
        F1mat = np.full((len(F1), cols), np.nan)
        for k in range(1, LG_MED + 1):
            # Lagged (past prepended with 1.0)
            F1mat[:, k - 1] = np.concatenate([np.ones(k), F1[:-k]])
            # Led (future appended with 1.0)
            F1mat[:, k + LG_MED - 1] = np.concatenate([F1[k:], np.ones(k)])
        F1mat[:, -1] = F1
        F1smooth = np.nanmin(F1mat, axis=1)

    return F1smooth, xi1_out
