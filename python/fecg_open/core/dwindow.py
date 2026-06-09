"""
dwindow.py — Python translation of dwindow.m in this library.

DWINDOW  Compute the derivative of a window function.

DH = DWINDOW(H) returns the derivative DH of the window vector H.

The derivative is computed by a centred-difference formula applied to
a zero-padded version of H, with endpoint corrections to account for
the step and ramp components of the window.

Example:
    from dwindow import dwindow
    from tftb_window import tftb_window

    h = tftb_window(1000)
    dh = dwindow(h)

Original algorithm: F. Auger (1994-1995).
"""

import numpy as np


def dwindow(h):
    """
    Compute the derivative of a window function.

    INPUT:
        h : array-like
            1D window vector (row or column)

    OUTPUT:
        Dh : ndarray
            Derivative of the window (same shape as input)

    """

    # Convert input to numpy array
    h = np.asarray(h, dtype=float)

    if h.size == 0:
        raise ValueError('one parameter required')

    if h.ndim == 1:
        hrow = h.shape[0]
        hcol = 1
        h = h.reshape(-1, 1)
    else:
        hrow, hcol = h.shape

    if hcol != 1:
        raise ValueError('h must have only one column)')
    h = h.flatten()

    # Lh may be a float when hrow is even (e.g. hrow=10 → Lh=4.5);
    # the ramp index vector must match this floating-point range.
    Lh = (hrow - 1) / 2

    if len(h) > hrow:
        step_height = (h[0] + h[hrow - 1]) / 2.0
        h_end = h[hrow - 1]
    else:
        step_height = (h[0] + h[-1]) / 2.0
        h_end = h[-1]

    if hrow == 1:
        ramp = 0.0
    else:
        ramp = (h_end - h[0]) / (hrow - 1)

    if Lh == int(Lh):
        ramp_indices = np.arange(-int(Lh), int(Lh) + 1)
    else:
        num_steps = int(2 * Lh + 1)
        ramp_indices = np.linspace(-Lh, Lh, num_steps)

    ramp_correction = ramp * ramp_indices

    if len(h) == hrow:
        h_corrected = h - step_height - ramp_correction
    else:
        h_corrected = h[:hrow] - step_height - ramp_correction

    # Zero-pad and apply centred difference to get the derivative
    h2 = np.concatenate([[0], h_corrected, [0]])
    Dh = (h2[2 : hrow + 2] - h2[0:hrow]) / 2.0 + ramp

    # Endpoint corrections for the step component
    if len(Dh) > 0:
        Dh[0] = Dh[0] + step_height
    if len(Dh) >= hrow:
        Dh[hrow - 1] = Dh[hrow - 1] - step_height

    return Dh


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dwindow.py <window_values>")
        print("\nExample:")
        print("  python dwindow.py -h")
        sys.exit(0)

    # Simple CLI for testing
    if sys.argv[1] == "-h":
        print(__doc__)
        sys.exit(0)

    try:
        window_values = np.array([float(x) for x in sys.argv[1:]])
        result = dwindow(window_values)
        print("Input: ", window_values)
        print("Output:", result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
