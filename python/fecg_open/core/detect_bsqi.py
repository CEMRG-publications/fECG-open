"""
detect_bsqi.py: SQI-based QRS detector orchestrator. 1:1 translation of detect_bsqi.m.

Computes a beat-by-beat Signal Quality Index (bSQI) for an ECG signal by comparing two
independent QRS detectors: jqrs (energy-based, via run_qrsdet_by_seg_ali) and a provided
reference annotation (acting as gqrs proxy). Returns the per-window F1 score.

INPUTS:
    data   — N×1 signal (ECG column vector)
    header — list of signal type labels, e.g. ['ECG']
    fs     — sampling rate (Hz)
    opt    — dict with SQI window/threshold parameters (see setDetectOptions defaults)
    beats  — 1×M integer, reference R-peak indices (gqrs proxy, 1-based samples)

OUTPUTS:
    qrs  — 1×K array, jqrs beat times (seconds)
    sqi  — list, per-window SQI vector (F1 scores, 0–1), one entry per signal

Only the ECG-only path is implemented (no ABP/PPG/SV blocks).
"""

import numpy as np
from .peak_detector.setDetectOptions import setDetectOptions
from .peak_detector.run_qrsdet_by_seg_ali import run_qrsdet_by_seg_ali
from .peak_detector.ecgsqi import ecgsqi


def detect_bsqi(data, header, fs, opt=None, beats=None):
    """
    SQI-based QRS detector. 1:1 translation of detect_bsqi.m (ECG-only path).

    For the fECG pipeline, this is called as:
        [~, sqi_f] = detect_bsqi(I2_orig', {'ECG'}, fs, opt, fbeats)
        bsqi_index = median(sqi_f{1})

    Parameters
    ----------
    data : ndarray
        Signal (N,) or (N, D)
    header : list
        Signal names, e.g. ['ECG']
    fs : int
        Sampling frequency
    opt : dict
        SQI options (see pipeline.m)
    beats : ndarray
        Pre-computed reference beat indices (1-based, sample units)

    Returns
    -------
    qrs : ndarray
        Final QRS beat times (seconds)
    sqi : list
        List of per-window SQI arrays (one per signal)
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    N, M = data.shape

    opt = setDetectOptions(opt)
    SIZE_WIND = opt['SIZE_WIND']
    LG_MED    = opt['LG_MED']
    REG_WIN   = opt['REG_WIN']
    THR       = opt['THR']
    JQRS_THRESH    = opt['JQRS_THRESH']
    JQRS_REFRAC    = opt['JQRS_REFRAC']
    JQRS_INTWIN_SZ = opt['JQRS_INTWIN_SZ']
    JQRS_WINDOW    = opt['JQRS_WINDOW']

    LG_REC = N / fs
    N_WIN = int(np.ceil(LG_REC / REG_WIN))

    # Identify ECG channels.
    # getSignalIndices is inlined here; only the ECG branch is implemented.
    # MATLAB getSignalIndices uses case-insensitive regex for 'ecg'; this matches
    # any header string containing 'ECG' (case-insensitive).
    idxECG = [i for i, h in enumerate(header) if 'ECG' in str(h).upper()]

    sqi_out = [None] * M

    for m in idxECG:
        # Run jqrs detector (MATLAB line 86)
        jqrs_opt = {
            'JQRS_THRESH': JQRS_THRESH,
            'JQRS_REFRAC': JQRS_REFRAC,
            'JQRS_INTWIN_SZ': JQRS_INTWIN_SZ,
            'JQRS_WINDOW': JQRS_WINDOW,
        }
        ann_jqrs = run_qrsdet_by_seg_ali(data[:, m], fs, jqrs_opt)

        # Reference annotation from supplied beats
        if beats is not None:
            beats_arr = np.asarray(beats).flatten()
            ann_gqrs = beats_arr[beats_arr > 0]
        else:
            ann_gqrs = np.array([])

        # Convert to seconds
        ann_jqrs_sec = ann_jqrs.astype(float) / fs
        ann_gqrs_sec = ann_gqrs.astype(float) / fs

        # Compute ecgsqi
        if len(ann_gqrs_sec) > 0 and len(ann_jqrs_sec) > 0:
            sqi_ecg, tsqi = ecgsqi(ann_gqrs_sec, ann_jqrs_sec,
                                    THR, SIZE_WIND, REG_WIN, LG_MED,
                                    LG_REC, N_WIN)
        else:
            sqi_ecg = np.zeros(N_WIN)

        sqi_out[m] = sqi_ecg

    # For single-channel ECG, return the jqrs detections and SQI
    if len(idxECG) > 0:
        qrs = ann_jqrs_sec if len(idxECG) > 0 else np.array([])
    else:
        qrs = np.array([])

    return qrs, sqi_out
