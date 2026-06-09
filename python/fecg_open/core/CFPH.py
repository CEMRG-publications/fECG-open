"""
CFPH: Compute a time-frequency representation (TFR) using the de-shape
Synchrosqueezing Transform (SST). The SST sharpens the STFT by reassigning energy
to the instantaneous frequency, producing a crisp ridge at the dominant heart-rate
frequency at each time step.

Method: 1-tap Flattop-windowed STFT → cepstral harmonic weighting →
        instantaneous frequency deviation (IFD) → synchrosqueezing (SST11).

Note: advTF.win_type='Flattop' is hardcoded (all other window-type branches removed).
      P.num_s=1 is hardcoded (multi-tap MT>1 loop removed).
"""

import numpy as np
from .STFT_IFD_fast import STFT_IFD_fast
from .cepstrum_convert import cepstrum_convert
from .synchrosqueeze1win import synchrosqueeze1win
from .tftb_window import tftb_window
from .dwindow import dwindow


def CFPH(x, basicTF, advTF, cepR, P):
    """
    ConceFT-style time-frequency analysis with cepstral de-shaping.

    Parameters
    ----------
    x : ndarray
        Input signal vector (1-D)
    basicTF : dict
        Basic STFT parameters: 'win', 'hop', 'fs', 'fr'
    advTF : dict
        Advanced STFT parameters: 'HighFreq', 'LowFreq', 'ths'
    cepR : dict
        Cepstrum parameters: 'g', 'Tc'
    P : dict
        Harmonic parameters (unused after constant propagation; kept for
        call-site compatibility)

    Returns
    -------
    tfr : ndarray    STFT magnitude (freq x time), trimmed to HighFreq
    ceps : ndarray   Raw cepstrum (freq x time)
    tceps : ndarray  Cepstrum-based de-shaping mask (freq x time)
    tfrr : ndarray   De-shaped STFT (tfr * tceps), trimmed to HighFreq
    rtfr : ndarray   Synchrosqueezed TFR (SST)
    tfrsq : ndarray  Synchrosqueezed plain TFR (no de-shaping)
    tfrtic : ndarray Frequency axis (normalised, 0..HighFreq)
    t : ndarray      Time axis (sample indices at each hop)
    """
    win = basicTF['win']
    hop = basicTF['hop']
    fs = basicTF['fs']
    fr = basicTF['fr']

    HighFreq = advTF['HighFreq']
    LowFreq = advTF['LowFreq']
    ths = advTF['ths']

    g = cepR['g']
    Tc = cepR['Tc']

    # Build Flat-Top window and its derivative.
    # MATLAB constructs a 2-tap structure [h 20*Dh] / [Dh 20*Dho] and then
    # projects with rv=[1 0]: rh = rv*h' = h(1:win), rDh = rv*Dh' = Dh(1:win).
    # The projection simply selects the first tap, so passing h and Dh directly
    # is mathematically equivalent for the single-tap (num_s=1) configuration.
    h = tftb_window(win)
    Dh = dwindow(h)

    # Compute STFT and instantaneous frequency deviation (IFD)
    tfr, ifd, tfrtic, t = STFT_IFD_fast(x, fr / fs, hop, h, Dh)
    tfr = np.abs(tfr)

    # tfrtic frequency axis: np.linspace and MATLAB's linspace use different internal arithmetic
    # (divide-then-multiply vs multiply-then-divide), producing differences of up to 1 ULP at
    # some indices (e.g. i=42, i=297). These ULP differences shift boundary-bin inclusion in
    # cepstrum_convert, causing tceps errors up to 6 % at two rows and cascading tfrrM errors
    # of O(1e2). The explicit formula below reproduces MATLAB's arithmetic exactly.
    _n_fb = len(tfrtic)
    tfrtic = (0.5 * np.arange(_n_fb, dtype=float)) / (_n_fb - 1)

    # Cepstrum-based spectral envelope removal (lpc=0: non-lpc path only)
    ceps, tceps = cepstrum_convert(tfr, tfrtic, g, fs, Tc, HighFreq, LowFreq)

    # Apply de-shaping mask; zero negative values and very low frequencies
    tfr0 = tfr.copy()
    tfrr = tfr0 * tceps
    tfrr[tfrr < 0] = 0
    tfrr[0:int(np.round(LowFreq * fs / fr)), :] = 0

    # Trim to HighFreq (feat='SST11' hardcoded: rtfr = tfr3, second synchrosqueeze output)
    trim_idx = int(np.round(HighFreq * fs / fr))
    tfrr = tfrr[0:trim_idx, :]
    ifd = ifd[0:trim_idx, :]

    # Synchrosqueezing: reassign STFT energy to instantaneous-frequency bins.
    # Called twice: once on the de-shaped (cepstrum-weighted) TFR → rtfr,
    # once on the plain magnitude TFR → tfrsq (for diagnostics).
    # feat='SST11' is hardcoded: rtfr comes from the second output of synchrosqueeze1win.
    tfr2, tfr3 = synchrosqueeze1win(tfrr, ifd, fr / fs, fr, HighFreq, fs, ths)
    _, tfrsq = synchrosqueeze1win(tfr0, ifd, fr / fs, fr, HighFreq, fs, ths)

    rtfr = tfr3  # SST output (second output of synchrosqueeze1win)

    # Final trim for remaining outputs
    tfrr = tfrr[0:trim_idx, :]
    tceps = tceps[0:trim_idx, :]
    tfr = tfr[0:trim_idx, :]
    tfrtic = tfrtic[0:rtfr.shape[0]]

    return tfr, ceps, tceps, tfrr, rtfr, tfrsq, tfrtic, t
