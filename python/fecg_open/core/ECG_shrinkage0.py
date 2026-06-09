"""
ECG_shrinkage0: Isolate the periodic (beat-aligned) ECG component using SVD-based
optimal shrinkage. Used both to extract the mECG template and to subtract it,
leaving the fECG residual.

Method: Build a beat matrix V (beat-window × beat-count); apply operator-norm optimal
        shrinkage to V's singular values; reconstruct with raised-cosine overlap-add
        weighting at beat boundaries.
        Window half-width = 95th percentile of RRI × 0.5.

Note: OptimalShrinkageOpt='op' (operator norm) is hardcoded; 'fro'/'nuc' branches removed.
"""

import numpy as np
from scipy import stats
from .optimal_shrinkage import optimal_shrinkage


def ECG_shrinkage0(x0, x0_real, current_beats, sigma_coeff, ifauto):
    """
    Optimal SVD shrinkage of a multi-beat ECG matrix.

    Parameters
    ----------
    x0 : ndarray
        Pre-processed ECG signal (1-D row vector)
    x0_real : ndarray
        Original unprocessed ECG signal (1-D row vector, same length as x0)
    current_beats : ndarray
        Sample indices of detected R-peaks (1-based, MATLAB convention)
    sigma_coeff : float
        Noise-level scaling coefficient
    ifauto : int
        If 1, auto-estimate noise; if 0, use fixed sigma=1

    Returns
    -------
    Om0 : ndarray
        Denoised version of x0 (1-D)
    Om0_real : ndarray
        Denoised version of x0_real (1-D)
    """

    x0 = np.asarray(x0).flatten()
    x0_real = np.asarray(x0_real).flatten()
    current_beats = np.asarray(current_beats).flatten().astype(int)

    # Compute beat-window dimensions from RR-interval distribution
    RRI = np.diff(current_beats)
    RRI = np.concatenate(([RRI[0]], RRI))  # prepend first interval

    # Use 95th-percentile RR as proxy for QT interval extent
    # CRITICAL: MATLAB prctile uses (i-0.5)/n cumulative probabilities,
    # numpy percentile uses p/100*(n-1) linear indexing — different results!
    # Must use MATLAB's method for exact match.
    RRI_sorted = np.sort(RRI)
    n_rri = len(RRI_sorted)
    prob = (np.arange(1, n_rri + 1) - 0.5) / n_rri
    pctile_95 = np.interp(0.95, prob, RRI_sorted)
    MaximalQTp = int(np.ceil(pctile_95 * 4 / 8))
    MaximalQTt = int(np.ceil(pctile_95 * 4 / 8))

    # =========================================================================
    # Trim beats that are too close to signal boundaries
    # =========================================================================
    tmp = np.where(current_beats > MaximalQTp)[0]
    current_beats = current_beats[tmp]
    RRI = RRI[tmp]

    tmp = np.where(current_beats + MaximalQTt <= len(x0))[0]
    current_beats = current_beats[tmp]
    RRI = RRI[tmp]

    if len(current_beats) == 0:
        return x0, x0_real

    # =========================================================================
    # Construct beat matrices V (pre-processed) and V2 (original)
    # =========================================================================
    V = []   # columns = individual beat waveforms from x0
    V2 = []  # columns = individual beat waveforms from x0_real
    II = []  # rows    = sample indices for each beat window

    for ii in range(len(current_beats)):
        # Sample-index range centred on this R-peak
        idx_start = int(current_beats[ii] - MaximalQTp - 1)
        idx_end = int(current_beats[ii] + MaximalQTt)

        if idx_start >= 0 and idx_end <= len(x0):
            V.append(x0[idx_start:idx_end])
            V2.append(x0_real[idx_start:idx_end])
            II.append(np.arange(idx_start, idx_end))

    if len(V) == 0:
        return x0, x0_real

    V = np.array(V).T   # shape: (window_size, num_beats)
    V2 = np.array(V2).T
    II = np.array(II)

    n_t, n_theta = V.shape

    # =========================================================================
    # Apply optimal SVD shrinkage to V (pre-processed signal)
    # =========================================================================

    # Noise estimate: sigma_coeff × RMS of the beat matrix after removing the
    # per-row (per-sample) median. This measures beat-to-beat variation rather
    # than absolute amplitude, making sigma adaptive to signal level.
    median_beat = np.median(V, axis=1, keepdims=True)
    sigma = sigma_coeff * np.sqrt(np.sum((V - median_beat)**2) / (n_t * n_theta))

    # Choose orientation so that beta = min(dim)/max(dim) <= 1
    if n_theta > n_t:
        beta0 = n_t / n_theta
        # SVD of normalized matrix
        U, S, Vt = np.linalg.svd(V / (sigma * np.sqrt(n_theta)), full_matrices=False)
        lambdaOS = S

        if ifauto:
            singvals_shrunk = optimal_shrinkage(lambdaOS, beta0)
        else:
            singvals_shrunk = optimal_shrinkage(lambdaOS, beta0, sigma=1)

        XN0 = sigma * np.sqrt(n_theta) * (U @ np.diag(singvals_shrunk) @ Vt[:n_t, :])
    else:
        beta0 = n_theta / n_t
        U, S, Vt = np.linalg.svd(V.T / (sigma * np.sqrt(n_t)), full_matrices=False)
        lambdaOS = S

        if ifauto:
            singvals_shrunk = optimal_shrinkage(lambdaOS, beta0)
        else:
            singvals_shrunk = optimal_shrinkage(lambdaOS, beta0, sigma=1)

        XN0 = (sigma * np.sqrt(n_t) * (U @ np.diag(singvals_shrunk) @ Vt[:n_theta, :])).T

    # =========================================================================
    # Apply optimal SVD shrinkage to V2 (original signal)
    # =========================================================================

    median_beat2 = np.median(V2, axis=1, keepdims=True)
    sigma = sigma_coeff * np.sqrt(np.sum((V2 - median_beat2)**2) / (n_t * n_theta))

    if n_theta > n_t:
        beta0 = n_t / n_theta
        U, S, Vt = np.linalg.svd(V2 / (sigma * np.sqrt(n_theta)), full_matrices=False)
        lambdaOS = S

        if ifauto:
            singvals_shrunk = optimal_shrinkage(lambdaOS, beta0)
        else:
            singvals_shrunk = optimal_shrinkage(lambdaOS, beta0, sigma=1)

        XN02 = sigma * np.sqrt(n_theta) * (U @ np.diag(singvals_shrunk) @ Vt[:n_t, :])
    else:
        beta0 = n_theta / n_t
        U, S, Vt = np.linalg.svd(V2.T / (sigma * np.sqrt(n_t)), full_matrices=False)
        lambdaOS = S

        if ifauto:
            singvals_shrunk = optimal_shrinkage(lambdaOS, beta0)
        else:
            singvals_shrunk = optimal_shrinkage(lambdaOS, beta0, sigma=1)

        XN02 = (sigma * np.sqrt(n_t) * (U @ np.diag(singvals_shrunk) @ Vt[:n_theta, :])).T

    # =========================================================================
    # Overlap-add reconstruction with raised-cosine tapering
    # =========================================================================

    Om0 = np.zeros(len(x0))
    Om0_real = np.zeros(len(x0_real))

    for s in range(len(current_beats)):
        XN_to_add = XN0[:, s]
        XN2_to_add = XN02[:, s]

        # Determine overlap lengths with neighbouring beats
        if s == 0:
            left_overlap = 2
            if s + 1 < len(current_beats):
                right_overlap = len(np.intersect1d(II[s+1, :], II[s, :]))
            else:
                right_overlap = 2
        elif s == len(current_beats) - 1:
            right_overlap = 2
            left_overlap = len(np.intersect1d(II[s-1, :], II[s, :]))
        else:
            left_overlap = len(np.intersect1d(II[s-1, :], II[s, :]))
            right_overlap = len(np.intersect1d(II[s+1, :], II[s, :]))

        # Guard against degenerate overlaps
        if left_overlap <= 1:
            left_overlap = 2
        if right_overlap <= 1:
            right_overlap = 2

        # Raised-cosine (Hann-like) taper over overlap regions
        window_len = MaximalQTp + MaximalQTt + 1
        W = np.ones(window_len)

        # Left taper
        left_taper = np.sin(np.linspace(0, np.pi/2, left_overlap))**2
        W[:left_overlap] = left_taper

        # Right taper (reversed: high to low from start to end of region)
        right_taper = np.sin(np.linspace(0, np.pi/2, right_overlap))**2
        W[-right_overlap:] = right_taper[::-1]

        # Apply taper and accumulate
        Om0_toadd = (W * XN_to_add)
        Om0_real_toadd = (W * XN2_to_add)

        idx_range = II[s, :]
        Om0[idx_range] += Om0_toadd
        Om0_real[idx_range] += Om0_real_toadd

    # =========================================================================
    # Fill any unprocessed samples (never covered by a beat window) with the
    # original signal. Also fills NaN values from numerical edge cases in the
    # SVD/shrinkage path.
    mask_Om0 = (Om0 == 0) | np.isnan(Om0)
    Om0[mask_Om0] = x0[mask_Om0]

    mask_Om0_real = (Om0_real == 0) | np.isnan(Om0_real)
    Om0_real[mask_Om0_real] = x0_real[mask_Om0_real]

    return Om0, Om0_real
