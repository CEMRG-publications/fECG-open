"""
ECG_shrinkage_median: Reconstruct a morphology-preserving fECG template using
nonlocal median shrinkage.

For each beat, the template is the median of that beat and its num_nonlocal nearest
neighbours in RRI space — beats with similar inter-beat intervals share similar
morphology and average out noise without blurring.

Method: Beat matrix + SVD optimal-shrinkage → nonlocal median template per beat →
        raised-cosine overlap-add reconstruction.
        Window half-width = 70th percentile of RRI × 0.5.

Note: OptimalShrinkageOpt='op' (operator norm) is hardcoded; 'fro'/'nuc' branches removed.
      Unprocessed samples are filled with LOESS-smoothed input (span=200),
      matching MATLAB's smooth(x0(Om0==0), 200, 'loess').
"""

import numpy as np
from scipy.interpolate import interp1d
from .optimal_shrinkage import optimal_shrinkage
from ..utils.matlab_smooth_wrapper import matlab_smooth


def ECG_shrinkage_median(x0, x0_real, current_beats, num_nonlocal, sigma_coeff, ifauto):
    """
    Non-local median optimal SVD shrinkage for ECG denoising.

    Parameters
    ----------
    x0 : ndarray
        Pre-processed ECG signal (1-D row vector)
    x0_real : ndarray
        Original unprocessed ECG signal (1-D row vector, same length as x0)
    current_beats : ndarray
        Sample indices of detected R-peaks (1-based, MATLAB convention)
    num_nonlocal : int
        Number of nearest-RR neighbours to include in median (including beat itself)
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

    # Compute beat-window half-widths from 70th-percentile RR interval
    RRI = np.diff(current_beats).astype(float)  # Convert to float to allow np.inf assignment
    RRI = np.concatenate(([RRI[0]], RRI))  # prepend first interval

    # Narrower window than ECG_shrinkage0 (0.70-quantile instead of 0.95)
    # CRITICAL: Use MATLAB's prctile method (i-0.5)/n, not numpy's default
    RRI_sorted = np.sort(RRI)
    n_rri = len(RRI_sorted)
    prob = (np.arange(1, n_rri + 1) - 0.5) / n_rri
    pctile_70 = np.interp(0.70, prob, RRI_sorted)
    MaximalQTp = int(np.ceil(pctile_70 * 4 / 8))
    MaximalQTt = int(np.ceil(pctile_70 * 4 / 8))

    # =========================================================================
    # Trim boundary beats
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
    V = []
    V2 = []
    II = []

    for ii in range(len(current_beats)):
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

    median_beat = np.median(V, axis=1, keepdims=True)
    sigma = sigma_coeff * np.sqrt(np.sum((V - median_beat)**2) / (n_t * n_theta))

    # Choose orientation so that beta = min(dim)/max(dim) <= 1
    if n_theta > n_t:
        beta0 = n_t / n_theta
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

    n_t, n_theta = V2.shape
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
    # Overlap-add reconstruction using non-local median
    # =========================================================================

    Om0 = np.zeros(len(x0))
    Om0_real = np.zeros(len(x0_real))

    for s in range(len(current_beats)):
        # Find the (num_nonlocal-1) beats with the most similar RR interval
        ZZZ = np.abs(RRI - RRI[s])
        ZZZ[s] = np.inf  # exclude the current beat from its own neighbourhood
        # numpy's default sort is not stable; on RR-distance ties the neighbour
        # set would silently differ, producing a different median waveform.
        # 'stable' ensures deterministic tie-breaking.
        sorted_idx = np.argsort(ZZZ, kind='stable')

        # Neighbourhood: beat s itself plus the closest num_nonlocal-1 beats
        Nidx = [s] + sorted_idx[:num_nonlocal-1].tolist()
        XN = XN0[:, Nidx]
        XN2 = XN02[:, Nidx]

        # Non-local median waveform
        XN_to_add = np.median(XN[:, :num_nonlocal], axis=1)
        XN2_to_add = np.median(XN2[:, :num_nonlocal], axis=1)

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

        # Right taper applies in reverse order (high-to-low amplitude toward signal boundary)
        right_taper = np.sin(np.linspace(0, np.pi/2, right_overlap))**2
        W[-right_overlap:] = right_taper[::-1]

        # Apply taper and accumulate
        Om0_toadd = W * XN_to_add
        Om0_real_toadd = W * XN2_to_add

        idx_range = II[s, :]
        Om0[idx_range] += Om0_toadd
        Om0_real[idx_range] += Om0_real_toadd

    # =========================================================================
    # Fill unprocessed samples with loess-smoothed version of input
    # =========================================================================

    # Fill unprocessed samples (those never covered by any beat window) with the
    # LOESS-smoothed input signal. Unlike ECG_shrinkage0 (which fills with the raw
    # signal), this function applies a LOESS smooth (span=200) before filling.
    unprocessed_idx_Om0 = np.where(Om0 == 0)[0]
    unprocessed_idx_Om0_real = np.where(Om0_real == 0)[0]

    if len(unprocessed_idx_Om0) > 0:
        try:
            unprocessed_segment = x0[unprocessed_idx_Om0]
            smoothed_segment = matlab_smooth(unprocessed_segment, span=200)
            Om0[unprocessed_idx_Om0] = smoothed_segment
        except Exception:
            # If smoothing raises (e.g. segment shorter than the span), fill with the raw signal.
            Om0[unprocessed_idx_Om0] = x0[unprocessed_idx_Om0]

    if len(unprocessed_idx_Om0_real) > 0:
        try:
            unprocessed_segment_real = x0_real[unprocessed_idx_Om0_real]
            smoothed_segment_real = matlab_smooth(unprocessed_segment_real, span=200)
            Om0_real[unprocessed_idx_Om0_real] = smoothed_segment_real
        except Exception:
            # If smoothing raises (e.g. segment shorter than the span), fill with the raw signal.
            Om0_real[unprocessed_idx_Om0_real] = x0_real[unprocessed_idx_Om0_real]

    return Om0, Om0_real
