"""
run_qrsdet_by_seg_ali.py: Segmented QRS detection wrapper. 1:1 translation of run_qrsdet_by_seg_ali.m.
"""

import numpy as np
from .qrs_detect2 import qrs_detect2


def run_qrsdet_by_seg_ali(ecg, fs=1000, opt=None):
    """
    Segmented QRS detection. 1:1 translation of run_qrsdet_by_seg_ali.m.

    Parameters
    ----------
    ecg : ndarray
        ECG signal (1-D)
    fs : int
        Sampling frequency
    opt : dict
        Options with keys: JQRS_WINDOW, JQRS_THRESH, JQRS_REFRAC, JQRS_INTWIN_SZ

    Returns
    -------
    QRS : ndarray
        Beat sample indices (1-based)
    """
    ecg = np.asarray(ecg, dtype=float).flatten()

    # Default options
    JQRS_THRESH = 0.3
    JQRS_WINDOW = 15
    JQRS_REFRAC = 0.250
    JQRS_INTWIN_SZ = 7

    if opt is not None:
        JQRS_THRESH = opt.get('JQRS_THRESH', JQRS_THRESH)
        JQRS_WINDOW = opt.get('JQRS_WINDOW', JQRS_WINDOW)
        JQRS_REFRAC = opt.get('JQRS_REFRAC', JQRS_REFRAC)
        JQRS_INTWIN_SZ = opt.get('JQRS_INTWIN_SZ', JQRS_INTWIN_SZ)

    segsizeSamp = int(JQRS_WINDOW * fs)
    NbSeg = len(ecg) // segsizeSamp
    QRS_segments = []
    signForce = 0

    if NbSeg == 0:
        return np.array([1])

    # First segment
    dTplus = int(fs)
    dTminus = 0
    start = 0  # 0-indexed
    stop = segsizeSamp - 1

    if NbSeg == 1:
        dTplus = 0
        stop = len(ecg) - 1

    seg = ecg[max(0, start - dTminus):min(len(ecg), stop + dTplus + 1)]
    QRStemp, signForce, _ = qrs_detect2(seg, JQRS_REFRAC, JQRS_THRESH, fs,
                                          None, signForce, 0, JQRS_INTWIN_SZ)
    # QRStemp is 0-indexed relative to seg start
    QRS_segments.append(QRStemp)

    start += segsizeSamp
    stop += segsizeSamp

    # Middle segments
    for ch in range(1, NbSeg - 1):
        dTplus = int(fs)
        dTminus = int(fs)

        seg_start = max(0, start - dTminus)
        seg_stop = min(len(ecg), stop + dTplus + 1)
        seg = ecg[seg_start:seg_stop]

        QRStemp, signForce, _ = qrs_detect2(seg, JQRS_REFRAC, JQRS_THRESH, fs,
                                              None, signForce, 0, JQRS_INTWIN_SZ)

        # Convert to global indices (0-indexed)
        NewQRS = seg_start + QRStemp
        NewQRS = NewQRS[(NewQRS >= start) & (NewQRS <= stop)]

        # Remove beats too close to previous segment's last detection
        if len(NewQRS) > 0 and len(QRS_segments) > 0:
            prev = QRS_segments[-1]
            if len(prev) > 0:
                NewQRS = NewQRS[NewQRS > prev[-1]]
                if len(NewQRS) > 0 and (NewQRS[0] - prev[-1]) < JQRS_REFRAC * fs:
                    NewQRS = NewQRS[1:]

        QRS_segments.append(NewQRS)
        start += segsizeSamp
        stop += segsizeSamp

    # Last segment
    if NbSeg > 1:
        stop = len(ecg) - 1
        dTplus = 0
        dTminus = int(fs)

        seg_start = max(0, start - dTminus)
        seg = ecg[seg_start:stop + 1]

        QRStemp, signForce, _ = qrs_detect2(seg, JQRS_REFRAC, JQRS_THRESH, fs,
                                              None, signForce, 0, JQRS_INTWIN_SZ)

        NewQRS = seg_start + QRStemp
        NewQRS = NewQRS[(NewQRS >= start) & (NewQRS <= stop)]

        if len(NewQRS) > 0 and len(QRS_segments) > 0:
            prev = QRS_segments[-1]
            if len(prev) > 0:
                NewQRS = NewQRS[NewQRS > prev[-1]]
                if len(NewQRS) > 0 and (NewQRS[0] - prev[-1]) < JQRS_REFRAC * fs:
                    NewQRS = NewQRS[1:]

        QRS_segments.append(NewQRS)

    # Concatenate all segments
    all_qrs = np.concatenate([q for q in QRS_segments if len(q) > 0])

    return all_qrs
