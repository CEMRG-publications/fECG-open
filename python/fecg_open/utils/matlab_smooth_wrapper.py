"""
matlab_smooth_wrapper.py: Exact MATLAB smooth(y, span, 'loess') — optimized.

For equally-spaced data, interior LOESS with fixed span reduces to convolution
with a precomputed kernel (since tricube weights and design matrix are constant).
Only edge points need the full least-squares solve.
"""

import numpy as np
from scipy.linalg import lstsq as scipy_lstsq

def _tricube(u):
    """Tricube weight: w(u) = (1 - |u|^3)^3 for |u| < 1, else 0."""
    u_abs = np.abs(u)
    return np.where(u_abs < 1.0, (1.0 - u_abs ** 3) ** 3, 0.0)

def _compute_loess_kernel(n_neighbors):
    """
    Precompute the LOESS convolution kernel for interior points.

    For equally-spaced data with fixed span, the degree-2 weighted least squares
    at the center point always yields: smoothed[i] = kernel @ x[i-half:i+half+1]
    """
    half = n_neighbors // 2

    # Positions relative to center (centered at 0)
    positions = np.arange(-half, half + 1, dtype=float) if n_neighbors % 2 == 1 \
        else np.arange(-half, half, dtype=float)
    actual_n = len(positions)

    # Distances and tricube weights (constant for interior points)
    distances = np.abs(positions)
    max_dist = distances.max()
    if max_dist > 0:
        normalized_dist = distances / max_dist
    else:
        normalized_dist = np.zeros(actual_n)
    wts = _tricube(normalized_dist)

    # Design matrix: [1, x, x^2]
    X = np.column_stack([np.ones(actual_n), positions, positions ** 2])

    # Weighted design matrix
    sqrt_wts = np.sqrt(wts)
    X_w = X * sqrt_wts[:, np.newaxis]

    # Solve for kernel: beta = (X_w'X_w)^-1 X_w' * diag(sqrt_wts)
    # At center (x=0), smoothed = beta[0] = first row of (X_w'X_w)^-1 X_w' @ diag(sqrt_wts)
    try:
        XwXw_inv = np.linalg.inv(X_w.T @ X_w)
        # kernel[j] = sum_k XwXw_inv[0,k] * X_w[j,k] * sqrt_wts[j]
        kernel = (XwXw_inv[0, :] @ X_w.T) * sqrt_wts
    except np.linalg.LinAlgError:
        kernel = wts / np.sum(wts)

    return kernel


def matlab_smooth(x, span):
    """
    Exact MATLAB smooth(y, span, 'loess') — optimized with convolution.

    Uses precomputed kernel for O(n) interior and Python loop only for edges.
    """
    x = np.asarray(x, dtype=float).flatten()
    n = len(x)

    if n < 3:
        return x.copy()

    # span must be odd; increment if even (matching smooth() behavior)
    n_neighbors = int(span)
    if n_neighbors % 2 == 0:
        n_neighbors += 1
    n_neighbors = max(3, min(n_neighbors, n))

    half_span = n_neighbors // 2

    # Precompute kernel for interior points
    kernel = _compute_loess_kernel(n_neighbors)

    # Interior: apply as convolution (numpy C code, very fast)
    smoothed = np.convolve(x, kernel[::-1], mode='same')

    # Fix edges: first and last half_span points need full LOESS solve
    # because the window size or position changes
    for i in list(range(half_span)) + list(range(n - half_span, n)):
        lo = max(0, i - half_span)
        hi = min(n - 1, i + half_span)

        if hi - lo + 1 < n_neighbors:
            if lo == 0:
                hi = min(n - 1, lo + n_neighbors - 1)
            elif hi == n - 1:
                lo = max(0, hi - n_neighbors + 1)

        idx = np.arange(lo, hi + 1)
        x_local = x[idx]
        x_pos = idx.astype(float)

        distances = np.abs(x_pos - float(i))
        max_dist = distances.max()
        normalized_dist = distances / max_dist if max_dist > 0 else np.zeros(len(idx))
        wts = _tricube(normalized_dist)

        if np.all(wts < 1e-15):
            smoothed[i] = x[i]
            continue

        x_centered = x_pos - float(i)
        X = np.column_stack([np.ones(len(idx)), x_centered, x_centered ** 2])
        sqrt_wts = np.sqrt(wts)
        X_w = X * sqrt_wts[:, np.newaxis]
        y_w = x_local * sqrt_wts

        try:
            beta, _, _, _ = scipy_lstsq(X_w, y_w)
            smoothed[i] = beta[0]
        except Exception:
            smoothed[i] = np.average(x_local, weights=wts)

    return smoothed
