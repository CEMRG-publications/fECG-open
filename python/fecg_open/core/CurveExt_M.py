"""
CurveExt_M: Extract the dominant frequency ridge from a time-frequency representation.

Uses dynamic programming with a temporal smoothness penalty.

Method: Convert TFR to negative log cost; forward DP minimises cost + λ·(Δfreq)²;
        traceback from minimum at final time frame.

Written by J. Lu. Optimized: Numba-JIT forward and backward passes.
"""

import numpy as np
from numba import njit

EPS_BACKWARD = 1e-8  # matches eps = 1e-8 in CurveExt_M.m

@njit(cache=True)
def _curve_ext_forward_backward(E, lambda_param, m, n):
    """
    Numba-JIT DP forward pass + MATLAB-style eps-scan backward pass.

    Forward pass: standard DP — FVal[ii,j] = min_k(FVal[ii-1,k] + λ*(k-j)²) + E[ii,j].

    Backward pass: MATLAB-style epsilon scan — at each frame ii, scans k=0..n-1 and
    picks the FIRST k whose cost is within eps of the accumulated minimum.  This
    matches MATLAB's "for kk=1:n; if abs(...) < eps; c(ii)=kk; break; end; end".

    Parameters
    ----------
    E : ndarray (m x n)
        Negative-log cost matrix (time x frequency).
    lambda_param : float
        Smoothness penalty weight.
    m : int
        Number of time frames.
    n : int
        Number of frequency bins.

    Returns
    -------
    c : ndarray (m,) int64
        Ridge curve as 1-based frequency-bin indices.
    FVal : ndarray (m x n) float64
        Accumulated DP cost matrix.
    """
    eps = 1e-8

    FVal = np.full((m, n), np.inf)
    FVal[0, :] = E[0, :]

    # Forward pass: fill FVal; no backlink storage needed
    for ii in range(1, m):
        for j in range(n):
            best = np.inf
            for k in range(n):
                cost = FVal[ii - 1, k] + lambda_param * (k - j) * (k - j)
                if cost < best:
                    best = cost
            FVal[ii, j] = best + E[ii, j]

    c = np.zeros(m, dtype=np.int64)

    best_val = np.inf
    for j in range(n):
        if FVal[m - 1, j] < best_val:
            best_val = FVal[m - 1, j]
            c[m - 1] = j + 1  # 1-based

    for ii in range(m - 2, -1, -1):
        curr_j = c[ii + 1] - 1  # 0-based current bin
        val = FVal[ii + 1, curr_j] - E[ii + 1, curr_j]
        found = False
        for kk in range(n):
            diff = val - FVal[ii, kk] - lambda_param * (kk - curr_j) * (kk - curr_j)
            if diff < 0.0:
                diff = -diff
            if diff < eps:
                c[ii] = kk + 1  # 1-based
                found = True
                break
        if not found:
            c[ii] = n // 2 + 1  # fallback: MATLAB uses round(n/2)

    return c, FVal


def CurveExt_M(P, lambda_param):
    """
    Extract a smooth ridge curve from a time-frequency energy map.

    Finds the frequency curve c(t) that maximises accumulated energy while
    penalising large frame-to-frame frequency jumps, via dynamic programming.

    Parameters
    ----------
    P : ndarray (time x freq)
    lambda_param : float
        Smoothness penalty weight.

    Returns
    -------
    c : ndarray (1-based frequency indices)
    FVal : ndarray (accumulated cost matrix)
    """
    eps = 1e-8

    E = P.astype(float)
    E = E / np.sum(E)
    E = -np.log(E + eps)

    m, n = E.shape

    # MATLAB: scans kk=1..n and picks the FIRST kk whose DP cost is within eps=1e-8 of
    #         the minimum — the lowest-indexed bin that qualifies
    # Python: explicit forward scan that returns the first index within eps=1e-8 of the
    #         per-row minimum, matching MATLAB's tie-breaking exactly
    c, FVal = _curve_ext_forward_backward(E, float(lambda_param), m, n)

    return c, FVal
