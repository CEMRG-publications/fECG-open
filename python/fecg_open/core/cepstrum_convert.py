"""
cepstrum_convert.py

CEPSTRUM_CONVERT  Map a time-frequency representation to a cepstral TF mask.

[ceps0, tceps] = CEPSTRUM_CONVERT(tfr, tfrtic, g, fs, Tc, HighFreq, LowFreq)

Converts each column of a TFR magnitude matrix to a real cepstrum,
then maps the cepstrum back onto the TF frequency axis to produce a
spectral-envelope mask (tceps). The mask is used to suppress
the spectral envelope of the ECG before synchrosqueezing, sharpening
harmonic peaks.

Algorithm overview:
  1. Compute real cepstrum: ifft(|tfr|^g).
  2. Zero out quefrency bins corresponding to frequencies above HighFreq.
  3. Truncate to the quefrency range corresponding to LowFreq.
  4. Interpolate the cepstrum and map each TF frequency bin to the
     sum of cepstral contributions from the matching quefrency range.
  5. Threshold at Tc.

INPUTS:
  tfr      - (F x T) TFR magnitude matrix
  tfrtic   - (F x 1) frequency axis in Hz for each TFR row
  g        - Cepstrum exponent: |tfr|^g
  fs       - Sampling frequency (Hz)
  Tc       - Minimum cepstral threshold
  HighFreq - Upper frequency limit (Hz)
  LowFreq  - Lower frequency limit (Hz)

OUTPUTS:
  ceps0  - Raw (unthresholded) cepstrum matrix (quefrency bins x T)
  tceps  - TF-mapped, thresholded cepstral mask (F x T)
"""

import numpy as np


def cepstrum_convert(tfr, tfrtic, g, fs, Tc, HighFreq, LowFreq):
    """
    Convert TFR to cepstral TF mask.

    INPUT:
        tfr : ndarray      (F x T) TFR magnitude matrix
        tfrtic : ndarray   (F,) frequency axis in Hz
        g : float          Exponent for |tfr|^g cepstrum
        fs : float         Sampling frequency (Hz)
        Tc : float         Cepstral threshold
        HighFreq : float   Upper frequency limit (Hz)
        LowFreq : float    Lower frequency limit (Hz)

    OUTPUT:
        ceps0 : ndarray    Raw cepstrum (quefrency_bins x T)
        tceps : ndarray    TF-mapped cepstral mask (F x T)
    """

    tfr = np.asarray(tfr)
    tfrtic = np.asarray(tfrtic).flatten()

    F, T = tfr.shape

    # Step 1: Compute power cepstrum (g != 0 always since g=0.3)
    ceps = np.real(
        np.fft.ifft(np.abs(tfr) ** g, n=2 * F, axis=0)
    )

    # Step 2: Zero out low quefrency bins (corresponding to > HighFreq)
    ceps[: round(1 / HighFreq), :] = 0

    # Remove NaN/Inf entries from log of zero-valued TFR bins
    ceps[np.isnan(ceps) | np.isinf(ceps)] = 0

    # Step 3: Truncate to the quefrency range of interest (down to LowFreq)
    ceps = ceps[: round(1 / LowFreq), :]
    ceps0 = ceps.copy()

    # Step 4: Interpolate and map cepstrum onto TF frequency axis (upsample 10x)
    num_quefrency = ceps.shape[0]
    num_interp = (num_quefrency - 1) * 10 + 1
    frac_idx = np.linspace(0, num_quefrency - 1, num_interp)
    lo = np.floor(frac_idx).astype(int)
    lo = np.clip(lo, 0, num_quefrency - 2)
    hi = lo + 1
    frac = frac_idx - lo
    ceps_interp = ceps[lo, :] * (1 - frac)[:, None] + ceps[hi, :] * frac[:, None]
    ceps = ceps_interp

    tceps = np.zeros((len(tfrtic), T))

    freq_scale = 10.0 * fs / np.arange(1, ceps.shape[0])

    n_tic = len(tfrtic)
    lower_bounds = (tfrtic[:-2] + tfrtic[1:-1]) * fs / 2
    upper_bounds = (tfrtic[2:] + tfrtic[1:-1]) * fs / 2

    # Each frequency bin is weighted equally (ii.^0 = 1), as adopted for the SampTA2017 manuscript.
    ceps_for_mapping = ceps[:len(freq_scale), :]
    for ii in range(n_tic - 2):
        lb = lower_bounds[ii]
        ub = upper_bounds[ii]
        mask = (freq_scale > lb) & (freq_scale < ub)
        if np.any(mask):
            tceps[ii + 1, :] = np.sum(ceps_for_mapping[mask, :], axis=0)

    # Step 5: Apply cepstral threshold
    tceps[tceps < Tc] = 0

    return ceps0, tceps


if __name__ == "__main__":
    print("cepstrum_convert module loaded.")
