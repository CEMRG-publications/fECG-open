"""
tftb_window.py — Python translation of tftb_window.m in this library.

TFTB_WINDOW  Generate a Flat-Top analysis window of length N.

The Flat-Top window coefficients used are:
    h = 0.2810639 + 0.5208972*cos(ind) + 0.1980399*cos(2*ind)
where ind = (-(N-1)/2 : (N-1)/2)' * 2*pi / (N-1).

Note: advTF.win_type='Flattop' is a constant in main; all other window-type
branches have been removed. This function always returns a Flattop window.

Original algorithm: F. Auger (1994-1995).

INPUT:
    N : int
        Length of window (positive integer)

OUTPUT:
    h : ndarray
        Window vector of shape (N,)
"""

import numpy as np


def tftb_window(N):
    """
    Generate a Flat-Top analysis window of length N.

    INPUT:
        N : int  Window length (positive)

    OUTPUT:
        h : ndarray  1-D array of length N
    """

    if N <= 0:
        raise ValueError('N should be strictly positive.')

    ind = np.arange(-(N - 1) / 2, (N + 1) / 2) * 2.0 * np.pi / (N - 1)
    h = (0.2810639
         + 0.5208972 * np.cos(ind)
         + 0.1980399 * np.cos(2.0 * ind))

    return h
