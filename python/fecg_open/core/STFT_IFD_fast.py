"""
STFT_IFD_fast: Short-Time Fourier Transform and Instantaneous Frequency Deviation (IFD).

Computes the STFT and the IFD needed for Synchrosqueezing. The IFD measures how much
each frequency bin's energy is offset from its nominal frequency — used by
synchrosqueeze1win to reassign STFT energy to sharper ridges.

Method: Frame signal with h; compute FFT;
        IFD = Im{ N * (Dh-STFT) / (h-STFT) / (2π) }

Reference: Synchrosqueezing by Li Su, 2015.
"""

import numpy as np


def STFT_IFD_fast(x, alpha, Hop, h, Dh):
    """
    Short-Time Fourier Transform with instantaneous frequency deviation.

    Computes the STFT and the first-order instantaneous frequency deviation
    (IFD) required for synchrosqueezing. The IFD at each TF point is
    estimated from the ratio of the derivative-window STFT to the analysis-
    window STFT (Daubechies, Lu & Wu 2011).

    Parameters
    ----------
    x : ndarray
        Input signal (column or row vector)
    alpha : float
        Frequency resolution: step size in normalised frequency (fr/fs)
    Hop : int
        Hop size in samples between successive frames
    h : ndarray
        Analysis window (column vector; will be transposed if necessary)
    Dh : ndarray
        Time-derivative of h (same orientation as h)

    Returns
    -------
    tfr : ndarray
        One-sided STFT magnitude (freq x time), size (N/2) x length(t)
    ifd : ndarray
        Instantaneous frequency deviation (same size as tfr), in
        normalised-frequency bins
    tfrtic : ndarray
        Normalised frequency axis (0 to 0.5), length = N/2
    t : ndarray
        Sample positions of frame centres (one per hop)
    """
    # Enforce column orientation for windows
    if h.ndim == 2 and h.shape[0] < h.shape[1]:
        h = h.T
    if Dh.ndim == 2 and Dh.shape[0] < Dh.shape[1]:
        Dh = Dh.T

    h = np.asarray(h).flatten()
    Dh = np.asarray(Dh).flatten()
    x = np.asarray(x).flatten()

    # Derived parameters
    # FFT length: use MATLAB's colon operator formula directly
    # N = length(-0.5+alpha : alpha : 0.5) in MATLAB
    # Match MATLAB's floating-point colon behavior: arange(-0.5+alpha, 0.5+0.1*alpha, alpha)
    N = len(np.arange(-0.5 + alpha, 0.5 + 0.1 * alpha, alpha))
    Win_length = len(h)
    Lh = (Win_length - 1) // 2  # half window length

    # Frequency axis (normalised, 0..0.5).
    # (N+1)//2 gives the correct bin count only for even N (standard pipeline: N=5000).
    # If N is ever odd, this must be revisited.
    n_freq_bins = (N + 1) // 2
    tfrtic = np.linspace(0, 0.5, n_freq_bins)

    n_frames = len(x) // Hop
    t = np.arange(1, n_frames + 1) * Hop

    # Frame extraction and windowing
    x_Frame = np.zeros((N, len(t)), dtype=complex)
    tf2 = np.zeros((N, len(t)), dtype=complex)

    n_half = (N + 1) // 2
    full_win_size = min(n_half - 1, Lh)
    tau_full = np.arange(-full_win_size, full_win_size + 1, dtype=int)
    circ_idx = (N + tau_full) % N
    h_win = np.conj(h[Lh + tau_full])
    Dh_win = np.conj(Dh[Lh + tau_full])
    norm_h_full = np.linalg.norm(h[Lh + tau_full])

    # Separate interior frames (full window fits) from boundary frames
    t_0based = t - 1
    interior_mask = (t_0based >= full_win_size) & (t_0based + full_win_size < len(x))

    if norm_h_full > 0 and np.any(interior_mask):
        interior_idx = np.where(interior_mask)[0]
        frame_centers = t_0based[interior_idx]
        # Gather all frame samples: (n_interior, win_len)
        sample_indices = frame_centers[:, np.newaxis] + tau_full[np.newaxis, :]
        x_segments = x[sample_indices]
        x_segments = x_segments - x_segments.mean(axis=1, keepdims=True)
        # Apply windows: (n_interior, win_len)
        windowed_h = x_segments * h_win[np.newaxis, :] / norm_h_full
        windowed_Dh = x_segments * Dh_win[np.newaxis, :] / norm_h_full
        # Place into output arrays using circular indices
        x_Frame[circ_idx[:, np.newaxis], interior_idx[np.newaxis, :]] = windowed_h.T
        tf2[circ_idx[:, np.newaxis], interior_idx[np.newaxis, :]] = windowed_Dh.T

    # Handle boundary frames with the original per-frame logic
    for ii in np.where(~interior_mask)[0]:
        ti_0based = t_0based[ii]
        max_left = min(n_half - 1, Lh, ti_0based)
        max_right = min(n_half - 1, Lh, len(x) - 1 - ti_0based)
        tau = np.arange(-max_left, max_right + 1, dtype=int)
        indices = (N + tau) % N
        norm_h = np.linalg.norm(h[Lh + tau])
        if norm_h == 0:
            continue
        x_segment = x[ti_0based + tau] - np.mean(x[ti_0based + tau])
        x_Frame[indices, ii] = x_segment * np.conj(h[Lh + tau]) / norm_h
        tf2[indices, ii] = x_segment * np.conj(Dh[Lh + tau]) / norm_h

    # FFT along frequency axis; keep only positive frequencies
    # Match MATLAB: tfr = tfr(1:round(N/2), :)
    tfr = np.fft.fft(x_Frame, n=N, axis=0)
    tfr = tfr[:n_freq_bins, :]

    tf2 = np.fft.fft(tf2, n=N, axis=0)
    tf2 = tf2[:n_freq_bins, :]

    # Estimate instantaneous frequency deviation
    # omega(k,n) = Im[N * Vdh(k,n) / Vh(k,n) / (2*pi)]
    # where Vh is the analysis STFT and Vdh is the derivative STFT.
    ifd = np.zeros_like(tfr, dtype=float)

    # Only compute where tfr is non-zero
    avoid_warn = tfr != 0
    ifd[avoid_warn] = np.imag(N * tf2[avoid_warn] / tfr[avoid_warn] / (2.0 * np.pi))

    # Return complex TFR (not magnitude), matching MATLAB output
    return tfr, ifd, tfrtic, t
