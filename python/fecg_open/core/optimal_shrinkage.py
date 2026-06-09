"""
optimal_shrinkage: Optimal hard-thresholding of singular values under operator (spectral)
norm loss, derived from the Gavish-Donoho (2014) asymptotically optimal shrinkage formula.

Shrinkage function: η(y) = max(x(y), 0) where
  x(y) = √[0.5·((y²−β−1) + √((y²−β−1)²−4β))] · 1(y ≥ 1+√β)

If sigma is omitted, it is estimated via the Marchenko-Pastur median.

Note: loss='op' (operator norm) is hardcoded; 'fro'/'nuc' branches and the
      OptimalShrinkageOpt parameter have been removed from all callers.

References:
  Gavish, M., & Donoho, D. L. (2014). The optimal hard threshold for singular values
  is 4/sqrt(3). IEEE Transactions on Information Theory, 60(8), 5040-5053.
"""

import numpy as np
from scipy.integrate import quad
import warnings


def optimal_shrinkage(singvals, beta, sigma=None):
    """
    Optimal operator-norm shrinkage of data singular values under white noise.

    Parameters
    ----------
    singvals : ndarray
        Vector of data singular values from SVD
    beta : float
        Aspect ratio m/n of the m-by-n data matrix (0 < beta <= 1)
    sigma : float, optional
        Noise standard deviation. If omitted, estimated from data via
        Marchenko-Pastur median.

    Returns
    -------
    singvals_shrunk : ndarray
        Optimally shrunken singular values (operator norm)
    """

    assert np.prod(np.shape(beta)) == 1, "beta must be scalar"
    assert 0 < beta <= 1, "beta must be in (0, 1]"
    assert np.prod(np.shape(singvals)) == len(singvals), "singvals must be 1-D"

    singvals = np.asarray(singvals).flatten()

    if sigma is None:
        warnings.filterwarnings('ignore')
        mp_median = _median_marchenko_pastur(beta)
        sigma = np.median(singvals) / np.sqrt(mp_median)
        warnings.filterwarnings('default')

    return _optshrink_impl(singvals, beta, sigma)


def _optshrink_impl(singvals, beta, sigma):
    """Operator-norm shrinkage via Baik-Ben Arous-Peche formula."""

    assert sigma > 0, "sigma must be positive"
    assert np.prod(np.shape(sigma)) == 1, "sigma must be scalar"

    def x_func(y):
        threshold = 1 + np.sqrt(beta)
        above_threshold = y >= threshold
        result = np.zeros_like(y, dtype=float)
        y_above = y[above_threshold]
        term = (y_above**2 - beta - 1)**2 - 4*beta
        term = np.maximum(term, 0)
        result[above_threshold] = np.sqrt(0.5 * ((y_above**2 - beta - 1) + np.sqrt(term)))
        return result

    def opt_op_shrink(y):
        return np.maximum(x_func(y), 0)

    return sigma * opt_op_shrink(singvals / sigma)


def _marchenko_pastur_density(x, beta):
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2
    if np.any((x < lobnd) | (x > hibnd)):
        return 0
    term = np.sqrt((hibnd - x) * (x - lobnd))
    return term / (2 * np.pi * beta * x)


def _inc_marpes(x0, beta, gamma=0):
    if beta > 1:
        raise ValueError('beta > 1 not supported')
    top_spec = (1 + np.sqrt(beta))**2

    def integrand(x):
        dens = _marchenko_pastur_density(x, beta)
        return (x**gamma) * dens if dens != 0 else 0

    try:
        I, _ = quad(integrand, x0, top_spec, limit=100)
    except:
        I = 0
    return I


def _median_marchenko_pastur(beta):
    lobnd = (1 - np.sqrt(beta))**2
    hibnd = (1 + np.sqrt(beta))**2

    def marches_cdf(x):
        return 1 - _inc_marpes(x, beta, gamma=0)

    change = True
    tolerance = 0.001
    max_iterations = 100
    iteration = 0

    while change and (hibnd - lobnd > tolerance) and iteration < max_iterations:
        iteration += 1
        change = False
        x_test = np.linspace(lobnd, hibnd, 5)
        y_test = np.array([marches_cdf(x) for x in x_test])

        below_half = y_test < 0.5
        above_half = y_test > 0.5

        if np.any(below_half):
            lobnd = np.max(x_test[below_half])
            change = True

        if np.any(above_half):
            hibnd = np.min(x_test[above_half])
            change = True

    return (hibnd + lobnd) / 2
