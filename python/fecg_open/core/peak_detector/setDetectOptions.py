"""
setDetectOptions: Validate and complete an options dict for the SQI/peak-detection pipeline.

Mirrors setDetectOptions.m from the peak-detector source.
"""


def setDetectOptions(opt=None):
    """
    Apply defaults and validate an SQI/peak-detection options dict.

    Parameters
    ----------
    opt : dict or None
        Partial options dict. Missing keys receive their default values.
        Unknown keys are ignored with a warning. None is treated as {}.

    Returns
    -------
    opt : dict
        Completed, validated options dict with all fields present.

    Fields
    ------
    SIZE_WIND : int (positive even)
        SQI evaluation window length (seconds). Default: 10.
    LG_MED : int (>= 0)
        Half-width of sliding-minimum smoothing across SQI windows. Default: 3.
    REG_WIN : int (>= 1)
        SQI re-evaluation stride (seconds). Default: 1.
    THR : float
        Beat-matching tolerance (seconds). Default: 0.150.
    SQI_THR : float in [0, 1]
        Minimum SQI to accept a signal as good quality. Default: 0.8.
    JQRS_THRESH : float
        jqrs energy threshold (fraction of signal max). Default: 0.3.
    JQRS_REFRAC : float
        jqrs refractory period (seconds). Default: 0.25.
    JQRS_INTWIN_SZ : int
        jqrs integration window size (samples). Default: 7.
    JQRS_WINDOW : int
        jqrs processing sub-window size (seconds). Default: 15.
    ENABLE_OTHER_DETECTORS : int (0 or 1)
        Enable ABP/PPG detectors. Default: 0.
    SAVE_STUFF : int (0 or 1)
        Save intermediate results to disk. Default: 0.
    HALF_WIND : float (derived)
        SIZE_WIND / 2. Set automatically; not a valid input field.
    """
    defaults = {
        'SIZE_WIND':               10,
        'LG_MED':                  3,
        'REG_WIN':                 1,
        'THR':                     0.150,
        'SQI_THR':                 0.8,
        'JQRS_THRESH':             0.3,
        'JQRS_REFRAC':             0.25,
        'JQRS_INTWIN_SZ':          7,
        'JQRS_WINDOW':             15,
        'ENABLE_OTHER_DETECTORS':  0,
        'SAVE_STUFF':              0,
    }

    if opt is None:
        opt = {}

    for key, val in list(opt.items()):
        if key not in defaults:
            print(f'setDetectOptions: ignoring unrecognised option {key!r}')
            continue

        if key in ('SAVE_STUFF', 'ENABLE_OTHER_DETECTORS'):
            if val not in (0, 1):
                raise ValueError(f'setDetectOptions: {key} must be 0 or 1, got {val!r}')

        elif key == 'SQI_THR':
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                raise ValueError(f'setDetectOptions: {key} must be in [0, 1], got {val!r}')

        elif key == 'SIZE_WIND':
            if not isinstance(val, int) or val <= 0 or val % 2 != 0:
                raise ValueError(
                    f'setDetectOptions: {key} must be a positive even integer, got {val!r}')

        elif key in ('LG_MED', 'REG_WIN', 'JQRS_INTWIN_SZ', 'JQRS_WINDOW'):
            if not isinstance(val, int):
                raise ValueError(f'setDetectOptions: {key} must be an integer, got {val!r}')

        elif key in ('THR', 'JQRS_THRESH', 'JQRS_REFRAC'):
            if not isinstance(val, (int, float)):
                raise ValueError(f'setDetectOptions: {key} must be numeric, got {val!r}')

        defaults[key] = val

    defaults['HALF_WIND'] = defaults['SIZE_WIND'] / 2
    return defaults
