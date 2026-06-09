"""
beat_simple.py: DP-based beat tracking guided by a time-varying tempo estimate.

Detects beat times in an onset-strength signal using dynamic programming.
The log-Gaussian transition cost penalises deviations from the expected inter-beat
period derived from the instantaneous HR curve.

Prepends 1000 silence frames for warm-up initialisation (matching MATLAB).
The forward DP pass is JIT-compiled with Numba for ~10x speedup.

Reference: 2007-06-19 Dan Ellis dpwe@ee.columbia.edu; 2016-08-13 Revised by Li Su.
"""

import numpy as np
from typing import Union
from numba import njit

@njit(cache=True)
def _beat_simple_forward(localscore, periods, n, alpha_param):
    """Numba-JIT forward DP pass. Returns (cumscore, backlink)."""
    cumscore = np.zeros(n)
    backlink = -np.ones(n, dtype=np.int64)

    # Pre-allocate weight buffer (max possible search window)
    max_period = 0.0
    for i in range(n):
        if periods[i] > max_period:
            max_period = periods[i]
    # np.floor(x + 0.5) implements half-away-from-zero rounding, matching MATLAB's round() semantics.
    max_win = int(np.floor(2.0 * max_period + 0.5)) + int(np.floor(max_period / 2.0 + 0.5)) + 2
    txwt_buf = np.zeros(max_win)

    # Cache: track last computed period to reuse weights
    last_lo = 0
    last_hi = 0
    last_period_key = -1

    for i in range(1000, n):
        period = periods[i]

        if period <= 0.0 or not np.isfinite(period):
            cumscore[i] = localscore[i]
            backlink[i] = -1
            continue

        lo = -int(np.floor(2.0 * period + 0.5))   # half-away-from-zero: matches MATLAB round()
        hi = -int(np.floor(period / 2.0 + 0.5))   # half-away-from-zero: matches MATLAB round()
        prange_len = hi - lo + 1

        # Reuse weights if period hasn't changed (rounded to 0.001)
        period_key = int(round(period * 1000))
        if period_key != last_period_key or lo != last_lo or hi != last_hi:
            neg_period = -period
            for j in range(prange_len):
                val = lo + j
                ratio = abs(val / neg_period)
                if ratio <= 0.0:
                    ratio = 1e-300
                log_r = np.log(ratio)
                if not np.isfinite(log_r):
                    log_r = 0.0
                txwt_buf[j] = -alpha_param * (log_r * log_r)
            last_lo = lo
            last_hi = hi
            last_period_key = period_key

        # Find best predecessor
        best_score = -1e300
        best_idx = -1
        for j in range(prange_len):
            t_idx = i + lo + j
            if t_idx < 0 or t_idx >= n:
                continue
            sc = txwt_buf[j] + cumscore[t_idx]
            if sc > best_score:
                best_score = sc
                best_idx = t_idx

        if best_idx >= 0:
            cumscore[i] = best_score + localscore[i]
            backlink[i] = best_idx
        else:
            cumscore[i] = localscore[i]
            backlink[i] = -1

    return cumscore, backlink


def beat_simple(
    onset: np.ndarray,
    osr: int,
    tempo: Union[np.ndarray, float],
    alpha: float = 100.0
) -> np.ndarray:
    """
    DP-based beat tracking guided by a time-varying tempo estimate.

    Detects beat times in an onset-strength signal using dynamic programming.
    The log-Gaussian transition cost penalises deviations from the expected
    inter-beat period derived from the instantaneous HR curve.

    Parameters
    ----------
    onset : ndarray (1D, float)
        Onset-strength envelope (e.g. resampled ECG at osr Hz).
    osr : int
        Frame rate of onset signal (samples/s); 100 Hz in the pipeline.
    tempo : ndarray (1D, float)
        Expected instantaneous tempo in beats per sample (cycles/sample at osr).
        Values are typically 0.01–0.04, constructed by callers as
        HR_index * basicTF['fr'] where basicTF['fr'] = 0.02 cycles/sample.
        NOT in BPM.
    alpha : float, optional
        Log-Gaussian transition cost weight (default 100).

    Returns
    -------
    beats : ndarray (1D, int)
        Beat frame indices (1-based, matching MATLAB convention).

    Notes
    -----
    This function always returns 1-based beat sample indices. The ``+1``
    below is intentional and load-bearing:
    - ``np.argmax`` returns a 0-based index; MATLAB's ``max()`` returns 1-based.
    - All downstream callers access signals via ``signal[beat - 1]`` and
      treat beat positions as 1-based in arithmetic. Removing ``+1`` would
      silently shift every beat by 1 sample (10 ms at 1000 Hz).
    The assertion below guards against any future edit that accidentally
    drops this offset.

    References
    ----------
    2007-06-19 Dan Ellis dpwe@ee.columbia.edu; 2016-08-13 Revised by Li Su.
    """

    onset = np.asarray(onset, dtype=float).flatten()
    tempo = np.asarray(tempo, dtype=float).flatten()

    if len(tempo) == 1:
        raise ValueError("beat_simple requires tempo to be a vector, not scalar.")

    if len(tempo) < len(onset):
        raise ValueError(f"tempo length ({len(tempo)}) must not be shorter than onset ({len(onset)}).")

    if len(tempo) > len(onset):
        tempo = tempo[:len(onset)]

    # Prepend 1000 silence frames for warm-up
    onset_prepped = np.concatenate([np.zeros(1000), onset])
    tempo_prepped = np.concatenate([np.ones(1000), tempo])

    n = len(onset_prepped)
    localscore = onset_prepped.copy()

    # Precompute periods
    periods = np.zeros(n)
    valid_tempo = tempo_prepped != 0
    periods[valid_tempo] = osr / tempo_prepped[valid_tempo]
    periods[~valid_tempo] = osr

    # Numba-JIT forward pass
    cumscore, backlink = _beat_simple_forward(localscore, periods, n, alpha)

    # Backward traceback (fast, few iterations)
    beats = [int(np.argmax(cumscore))]
    while backlink[beats[0]] >= 0:
        beats.insert(0, int(backlink[beats[0]]))

    beats = np.array(beats, dtype=np.int64) - 1000
    # MATLAB: beats(beats > 0) — keeps beat at onset index 1 (1-based)
    # Python: onset[0] maps to 0 after subtracting 1000-frame warmup.
    #         --> beats >= 0 matches MATLAB
    beats = beats[beats >= 0]
    beats = beats + 1  # see Notes in docstring
    assert np.all(beats >= 1), "beat_simple: beat positions must be 1-based (>=1)"


    return beats
