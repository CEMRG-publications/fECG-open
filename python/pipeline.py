import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt
from scipy.interpolate import PchipInterpolator, interp1d
import time
import os
import h5py
import hdf5storage
import argparse

from fecg_open.utils.MyReadDataq_32 import MyReadDataq_32
from fecg_open.utils.parsave_output_file import parsave_output_file
from fecg_open.core.ECG_detrend import ECG_detrend
from fecg_open.core.mbeats_extract_real import mbeats_extract_real
from fecg_open.core.fbeats_extract_real import fbeats_extract_real
from fecg_open.core.fusion_mbeats import fusion_mbeats
from fecg_open.core.mbeats_modify import mbeats_modify
from fecg_open.core.ECG_shrinkage0 import ECG_shrinkage0
from fecg_open.core.ECG_shrinkage_median import ECG_shrinkage_median
from fecg_open.core.detect_bsqi import detect_bsqi
from fecg_open.utils.align_beats_to_ecg import align_beats_to_ecg
from fecg_open.utils.pan_tompkin_revised import pan_tompkin_revised

def _load_partial_results(path):
    """
    Load previously saved segment data from an output file for mid-file resume.

    Returns a dict mapping each final_* list name to a Python list populated
    from the HDF5 file.
    """
    data = hdf5storage.loadmat(path)

    def _cell_list(key):
        arr = data.get(key, np.empty((1, 0), dtype=object))
        return list(np.asarray(arr).flatten())

    def _row_list(key):
        arr = data.get(key, np.empty((1, 0), dtype=object))
        flat = np.asarray(arr).flatten()
        return [float(np.asarray(x).flat[0]) for x in flat]

    loaded = {
        'final_file_names':  [str(s) for s in _cell_list('final_file_names')],
        'final_start_times': _row_list('final_start_times'),
        'final_og_ECG':      _cell_list('final_og_ECG'),
        'final_SQI':         _row_list('final_SQI'),
        'final_chs_used':    _cell_list('final_chs_used'),
        'final_Om':          _cell_list('final_Om'),
        'final_Of':          _cell_list('final_Of'),
        'final_aECG':        _cell_list('final_aECG'),
        'final_mbeats':      _cell_list('final_mbeats'),
        'final_fbeats':      _cell_list('final_fbeats'),
        'final_proc_time':   _row_list('final_proc_time'),
    }
    return loaded

def _make_filter_stage(order, wn, btype):
    """
    Return (sos, matlab_padlen) for one filter stage.
    """
    b, a = butter(order, wn, btype)
    sos  = butter(order, wn, btype, output='sos')
    return sos, 3 * (max(len(b), len(a)) - 1)

def main():
    """
    Run fECG decomposition pipeline.
    """

    # ============================== Configuration ================================
    parser = argparse.ArgumentParser(
        description="Process four-channel abdominal fECG recordings."
    )

    parser.add_argument(
        '-i',
        '--input',
        help="Path to the input folder containing channel files",
        dest="input_folder",
    )

    parser.add_argument(
        '-o',
        '--output',
        help="Path to the output folder where results will be saved",
        dest="output_folder",
    )

    args = parser.parse_args()

    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")

    input_folder = args.input_folder
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    # --- Signal constants --------------------------------------------------------
    NUM_CHANNELS = 4
    fs = 1000

    # ============================== SQI Parameters ==============================
    # bSQI = beat-by-beat Signal Quality Index. Scores 0–1 how well two
    # independent QRS detectors agree. A score > SQI_THR (0.8) is "good".
    opt = {
        'SIZE_WIND': 4,       # window size (s) for each bSQI evaluation window
        'LG_MED': 0,          # median smoothing across adjacent SQI windows (0 = off)
        'REG_WIN': 1,         # how often (s) SQI is re-evaluated
        'THR': 0.150,         # tolerance (s) for calling two detected beats a match
        'SQI_THR': 0.8,       # minimum SQI to consider a signal "good quality"
        'JQRS_THRESH': 0.3,   # jqrs energy threshold (relative to signal max)
        'JQRS_REFRAC': 0.25,  # jqrs refractory period (s) — minimum inter-beat gap
        'JQRS_INTWIN_SZ': 7,  # jqrs integration window size (samples at fs)
        'JQRS_WINDOW': 15,    # jqrs processing sub-window size (s)
    }

    # ============================== De-shape STFT (TF analysis) parameters ======
    # These control the Synchrosqueezing Transform used to estimate instantaneous
    # heart rate as a ridge in the time-frequency plane.
    basicTF = {
        'win': 1000,   # STFT window length (samples at 100 Hz internal rate = 10 s)
        'hop': 20,     # STFT hop size (samples) — controls time resolution
        'fs': 100,     # internal resampled rate (Hz) used inside the TF pipeline
        'fr': 0.02,    # frequency resolution (cycles per sample at basicTF.fs)
    }

    advTF = {
        'ths': 1e-6,             # synchrosqueezing threshold — suppress near-zero TFR entries
        'HighFreq': 10 / 100,    # upper heart rate limit (fraction of basicTF.fs = 10 Hz → 600 bpm)
        'LowFreq': 0.5 / 100,   # lower heart rate limit (0.5 Hz → 30 bpm)
    }

    cepR = {'g': 0.3, 'Tc': 0}  # g: cepstral exponent; Tc: threshold (zero values below this level)
    P = {'num_s': 1}  # num_s: number of synchrosqueeze harmonics (1 = fundamental only)

    alphaN = 7 # half-width of the α grid; generates 2*alphaN=14 channel-mix orientations

    # ============================== Morphology / shrinkage parameters ============
    num_med = 51        # short median filter length (samples) for detrending without morphology
    num_med_l = 301     # long median filter length — suppresses low-frequency drift
    num_med_s = 101     # short median filter for morphology-preserving detrend
    num_nonlocal = 10   # number of nearest-RRI neighbours used in nonlocal median shrinkage

    # ============================== Precomputed constants =======================

    # 14 orientation weights α ∈ [-1,1]: x0 = √(1-α²)·sig1 + α·sig2 gives a
    # unit-norm linear combination of two channels.
    alpha_ind = (np.arange(1, alphaN * 2 + 1) - alphaN) / alphaN

    # --- Filter design -------------------------------------------------------
    # Filters are constructed in SOS (second-order sections) form via
    # butter(..., output='sos') rather than the polynomial BA form used in
    # the MATLAB reference. This is necessary for three reasons:
    #
    # (i)  Coefficient precision: the 5th-order bandstop (effective order 10)
    #      requires representing a degree-10 polynomial whose roots lie within
    #      ~2 Hz of each other near 50 Hz. Converting those roots to polynomial
    #      coefficients introduces floating-point rounding of ~1e-13, which
    #      propagates through the filter chain. SOS keeps each stage at most
    #      degree 2, avoiding this accumulation.
    #
    # (ii) MATLAB parity: MATLAB filtfilt (>= R2014b) converts BA to SOS
    #      internally before filtering. Reproducing that numerical path in
    #      Python requires constructing SOS directly.
    #
    # (iii) Padlen coupling: scipy sosfiltfilt does not infer padlen from the
    #       original filter order. The MATLAB-equivalent padlen is therefore
    #       derived from the BA form at design time via
    #       padlen = 3 * (max(len(b), len(a)) - 1), and stored alongside the
    #       SOS array. See _make_filter_stage() for the implementation.

    # Filters for channels 1-3 (fs_og = 300 Hz; Nyquist = 150 Hz)
    fs_og = 300
    sos_lp300, pl_lp300 = _make_filter_stage(5, 120 / (fs_og/2), 'low')
    sos_hp300, pl_hp300 = _make_filter_stage(5, 0.5 / (fs_og/2), 'high')
    sos_bs300, pl_bs300 = _make_filter_stage(5, np.array([48, 52]) / (fs_og/2), 'bandstop')

    # Filters for channel 4 (fs_og = 900 Hz; Nyquist = 450 Hz)
    fs_og = 900
    sos_lp900, pl_lp900 = _make_filter_stage(5, 120 / (fs_og/2), 'low')
    sos_hp900, pl_hp900 = _make_filter_stage(5, 0.5 / (fs_og/2), 'high')
    sos_bs900, pl_bs900 = _make_filter_stage(5, np.array([48, 52]) / (fs_og/2), 'bandstop')

    # ============================== Input File Discovery ========================
    DB_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.ch1')])

    # ============================== Main File Loop ==============================

    for file_i, db_file in enumerate(DB_files):

        start_clock = time.time()

        # --- Initialise per-file output cell arrays ------------------------------
        final_file_names = []
        final_start_times = []
        final_og_ECG = []
        final_SQI = []
        final_chs_used = []
        final_Om = []
        final_Of = []
        final_aECG = []
        final_mbeats = []
        final_fbeats = []
        final_proc_time = []

        # Per-channel signal storage (preprocessed)
        og_ecg = [None] * (NUM_CHANNELS + 1)       # 1-indexed: [None, ch1, ch2, ch3, ch4]
        x_up_c = [None] * (NUM_CHANNELS + 1)
        x_up_c_real = [None] * (NUM_CHANNELS + 1)

        # --- Print progress -------------------------------------------------------
        file_name = db_file.replace('.ch1', '')
        print(f'Loading [{file_i + 1}/{len(DB_files)}]: {file_name}...')

        # --- Initialise output file (or load partial results for resume) ---------
        #
        # Resume logic:
        #   When an output file already exists, int(_f['final_file_names'].size)
        #   read to determine how many segments were previously completed. The
        #   completed segment data is then loaded back into the final_* lists so
        #   that the trailing parsave call (inside the segment loop) always writes
        #   the full accumulated result. This matches MATLAB's per-segment matfile()
        #   write: each segment is durably saved as it completes, so a crash does
        #   not force reprocessing of already-finished segments.
        output_file_path = os.path.join(output_folder, file_name + '.mat')
        segment_offset = 0
        time_offset = 0
        last_fbeats_len = float('nan')

        if os.path.isfile(output_file_path):
            try:
                with h5py.File(output_file_path, 'r') as _f:
                    segment_offset = int(_f['final_file_names'].size)
                if segment_offset > 0:
                    prev = _load_partial_results(output_file_path)
                    final_file_names  = prev['final_file_names']
                    final_start_times = prev['final_start_times']
                    final_og_ECG      = prev['final_og_ECG']
                    final_SQI         = prev['final_SQI']
                    final_chs_used    = prev['final_chs_used']
                    final_Om          = prev['final_Om']
                    final_Of          = prev['final_Of']
                    final_aECG        = prev['final_aECG']
                    final_mbeats      = prev['final_mbeats']
                    final_fbeats      = prev['final_fbeats']
                    final_proc_time   = prev['final_proc_time']
                    time_offset = final_proc_time[-1] if final_proc_time else 0.0
                    last_fbeats_len = float(len(final_fbeats[-1])) if final_fbeats else float('nan')
                print(f'Resuming from segment {segment_offset} (output file exists)')
            except Exception as _e:
                print(f'Warning: cannot read existing output file ({_e}); reprocessing from start')
                segment_offset = 0
        else:
            segment_offset = 0
            # Create initial empty output file before the loop begins
            parsave_output_file(
                output_file_path,
                final_file_names, final_start_times, final_og_ECG,
                final_SQI, final_chs_used, final_Om,
                final_Of, final_aECG, final_mbeats,
                final_fbeats, final_proc_time)

        # ========================= Preprocessing (all channels) =================

        for ch_i in range(1, NUM_CHANNELS + 1):
            channel_file_path = os.path.join(input_folder, f'{file_name}.ch{ch_i}')

            # Read raw int32 samples and remove DC offset
            channel_signal = MyReadDataq_32(channel_file_path)
            channel_signal = channel_signal - np.mean(channel_signal)
            channel_signal = channel_signal.flatten()  # ensure 1-D (row vector)

            # Channels 1-3 are 300 Hz; channel 4 is 900 Hz
            if ch_i < 4:
                fs_og = 300
                sos_low,  pl_low  = sos_lp300, pl_lp300
                sos_high, pl_high = sos_hp300, pl_hp300
                sos_notch, pl_notch = sos_bs300, pl_bs300
            else:
                fs_og = 900
                sos_low,  pl_low  = sos_lp900, pl_lp900
                sos_high, pl_high = sos_hp900, pl_hp900
                sos_notch, pl_notch = sos_bs900, pl_bs900

            print(f'Preprocessing [{ch_i}/{NUM_CHANNELS}]: {file_name}.ch{ch_i}...')

            # --- Bandpass + notch filtering (in original sample rate domain) -----
            channel_signal = channel_signal - np.mean(channel_signal)

            # --- Filtering -----------------------------------------------------------
            # sosfiltfilt is called with:
            #   padtype='odd'  - anti-symmetric reflection, matching MATLAB filtfilt default.
            #   padlen=pl_*    - per-filter MATLAB convention: 3*(max(len(b),len(a))-1).
            #                    scipy's default (3*max(len(b),len(a))) adds 3 extra samples
            #                    per stage; empirical testing showed this raises interior RMSE
            #                    by three to four orders of magnitude on all channels.
            channel_signal = sosfiltfilt(sos_low, channel_signal, padtype='odd', padlen=pl_low)
            channel_signal = sosfiltfilt(sos_high, channel_signal, padtype='odd', padlen=pl_high)
            channel_signal = sosfiltfilt(sos_notch, channel_signal, padtype='odd', padlen=pl_notch)

            # --- Scale and resample to 1000 Hz via pchip interpolation -----------
            channel_signal = channel_signal / 1000.0

            n_samples = len(channel_signal)
            orig_time = np.linspace(0, n_samples / fs_og, n_samples)
            new_time = np.linspace(0, orig_time[-1], int(np.round(orig_time[-1] * fs)))
            pchip = PchipInterpolator(orig_time, channel_signal)
            channel_signal = pchip(new_time)

            # --- Store original interpolated signal (before detrend) -------------
            og_ecg[ch_i] = channel_signal.copy()

            # --- Baseline wander removal -----------------------------------------
            x_up_c[ch_i] = ECG_detrend(channel_signal, num_med, num_med_l, 0)
            x_up_c_real[ch_i] = ECG_detrend(channel_signal, num_med_s, num_med_l, 1)

        # ========================= Segmentation =================================

        full_length = len(x_up_c[1])
        full_length = full_length - (full_length % fs)

        SEGMENT_TIME_RANGES = list(range(0, full_length + 1, 60 * fs))
        if full_length - SEGMENT_TIME_RANGES[-1] > 0:
            SEGMENT_TIME_RANGES.append(full_length)
        NUM_SEGMENTS = len(SEGMENT_TIME_RANGES) - 1

        if segment_offset > 0:
            print(f'Resuming from segment {segment_offset} of {NUM_SEGMENTS}')

        # ========================= Decomposition Loop ===========================

        for seg_i in range(segment_offset, NUM_SEGMENTS):

            # DP smoothness weights — loosened here, tightened in the refinement pass
            lam_curve = 50  # TF ridge extraction penalty (higher → smoother HR curve)
            lam_beat = 50   # beat-tracking transition penalty (higher → more regular rhythm)

            start_ind = SEGMENT_TIME_RANGES[seg_i] + 1      # 1-indexed
            end_ind = SEGMENT_TIME_RANGES[seg_i + 1]         # 1-indexed

            # +/- 1s padding, clipped at signal boundaries
            start_ind_padded = max(start_ind - fs, 1)
            end_ind_padded = min(end_ind + fs, full_length)

            print(f'Decomposing [{seg_i + 1}/{NUM_SEGMENTS}]: {file_name} '
                  f'[{start_ind_padded}:{end_ind_padded}]...')

            # =================== Channel Pair Selection =========================

            bsqi_index_max = -np.inf
            chs_used = []

            # Variables to store best combination results
            alpha_current = None
            fbeats_current = None
            mbeats_current = None
            x0_current = None
            x0_real_current = None
            I2_orig_current = None
            I2_orig_real_current = None
            Om_real_current = None
            Om_current = None

            for ch_i in range(1, NUM_CHANNELS):
                for ch_i2 in range(ch_i + 1, NUM_CHANNELS + 1):

                    # Extract padded segment for this channel pair
                    sig1 = x_up_c[ch_i][start_ind_padded - 1:end_ind_padded]
                    sig2 = x_up_c[ch_i2][start_ind_padded - 1:end_ind_padded]
                    sig1_real = x_up_c_real[ch_i][start_ind_padded - 1:end_ind_padded]
                    sig2_real = x_up_c_real[ch_i2][start_ind_padded - 1:end_ind_padded]

                    # --- Maternal beat detection across all alpha combinations ---
                    alpha_sig_c = [None] * len(alpha_ind)
                    alpha_mbeats_c = [None] * len(alpha_ind)
                    alpha_hrv_c = np.full(len(alpha_ind), np.inf)
                    x0_raw_c = [None] * len(alpha_ind)
                    x0_real_raw_c = [None] * len(alpha_ind)

                    for i in range(len(alpha_ind)):
                        alpha = alpha_ind[i]
                        w1 = np.sqrt(1 - alpha ** 2)

                        x0_raw_c[i] = w1 * sig1 + alpha * sig2
                        x0_real_raw_c[i] = w1 * sig1_real + alpha * sig2_real

                        x0 = x0_raw_c[i]
                        x0_real = x0_real_raw_c[i]

                        try:
                            x0_out, _, mbeats, _, _, _, _, _, _, _ = \
                                mbeats_extract_real(x0, x0_real, fs, basicTF, advTF,
                                                    cepR, P, lam_curve, lam_beat)
                        except Exception:
                            continue

                        alpha_sig_c[i] = x0_out
                        alpha_mbeats_c[i] = mbeats

                        if len(mbeats) > 2:
                            # RMSSD-like HRV: low value → regular rhythm → good maternal ECG orientation
                            hrv = np.diff(np.diff(mbeats.astype(float)))
                            alpha_hrv_c[i] = np.sqrt(np.sum(hrv ** 2) / len(hrv))

                    # Select the 5 most regular (lowest HRV) orientations for fusion
                    sorted_idx = np.argsort(alpha_hrv_c)
                    top5_idx = sorted_idx[:5]

                    # Filter out invalid entries (where extraction failed)
                    top5_idx = [idx for idx in top5_idx
                                if alpha_sig_c[idx] is not None and
                                alpha_mbeats_c[idx] is not None]
                    if not top5_idx:
                        continue

                    asig_c = [alpha_sig_c[idx] for idx in top5_idx]
                    ambeats_c = [alpha_mbeats_c[idx] for idx in top5_idx]

                    # --- Fuse maternal beats from the top 5 combinations ---------
                    try:
                        mbeats_fus = fusion_mbeats(ambeats_c, asig_c, fs)
                    except Exception:
                        continue

                    if len(mbeats_fus) > 20:
                        # Convert fused beat times to an interpolated HR vector (bpm)
                        # sampled every 200 ms — used to suppress maternal HR in the fetal TFR
                        RRI = np.diff(mbeats_fus.astype(float))
                        RRI = np.concatenate([[RRI[0]], RRI])
                        x0_len = len(x0_raw_c[top5_idx[0]])
                        query_pts = np.arange(1, x0_len + 1, 200, dtype=float)
                        f_hr = interp1d(mbeats_fus.astype(float),
                                        60.0 * fs / RRI,
                                        kind='nearest',
                                        fill_value='extrapolate',
                                        bounds_error=False)
                        HR_ma_fus = f_hr(query_pts)
                        HR_ma_fus = np.round(HR_ma_fus).astype(int)
                        HR_ma_fus[HR_ma_fus < 0] = 10  # clamp negative artefacts
                    else:
                        # Fallback if fusion fails: use beats from the single best orientation
                        print('bad fusion result, use channel 1 R peaks')
                        try:
                            sig1_out, _, mbeats_p1, _, _, _, _, HR_ma_fus, _, _ = \
                                mbeats_extract_real(sig1, sig1_real, fs, basicTF, advTF,
                                                    cepR, P, lam_curve, lam_beat)
                            mbeats_fus = mbeats_p1
                        except Exception:
                            continue

                    # --- Estimate fECG and compute bSQI for each alpha -----------
                    # Evaluate all orientations (not just the top 5 used for fusion)
                    # to find the bSQI-maximising combination.
                    for i in range(len(alpha_ind)):
                        if x0_raw_c[i] is None:
                            continue

                        x0 = x0_raw_c[i]
                        x0_real = x0_real_raw_c[i]

                        # Relocate fused maternal beats to local peaks
                        try:
                            mbeats_loc, _, _, _, _ = mbeats_modify(x0, mbeats_fus)
                        except Exception:
                            continue

                        # First-pass maternal ECG extraction by SVD shrinkage
                        try:
                            Om, Om_real = ECG_shrinkage0(x0, x0_real, mbeats_loc,
                                                         1.5, 0)
                        except Exception:
                            print('Decomposition failed due to high signal noise. '
                                  'Skipping...')
                            Om = np.zeros(len(x0))
                            Om_real = np.zeros(len(x0_real))

                        # Rough fetal residual
                        I2_orig_real = x0_real - Om_real
                        I2_orig = x0 - Om

                        # Fetal beat detection in residual
                        try:
                            _, _, fbeats, _, _, _, _, _, _, _ = \
                                fbeats_extract_real(I2_orig, I2_orig_real, fs,
                                                    basicTF, advTF, cepR, P,
                                                    lam_curve, lam_beat,
                                                    HR_ma_fus)
                        except Exception:
                            continue
                        if len(fbeats) == 0:
                            continue

                        # Compute beat SQI
                        try:
                            _, sqi_f = detect_bsqi(
                                I2_orig, ['ECG'], fs, opt, fbeats.astype(int))
                            sqi_values = sqi_f[0] if sqi_f[0] is not None else np.array([0.0])
                            bsqi_index = np.median(sqi_values) if len(sqi_values) > 0 else 0.0
                        except Exception:
                            bsqi_index = 0.0

                        # Keep this combination if it has the best bSQI so far
                        if bsqi_index > bsqi_index_max:
                            print(f'Found channels: [{ch_i}, {ch_i2}] '
                                  f'w/ higher bSQI: {bsqi_index:6.3f}')

                            alpha_current = alpha_ind[i]
                            bsqi_index_max = bsqi_index
                            chs_used = [ch_i, ch_i2]
                            mbeats_current = mbeats_loc
                            fbeats_current = fbeats
                            x0_current = x0
                            x0_real_current = x0_real
                            I2_orig_current = I2_orig
                            I2_orig_real_current = I2_orig_real
                            Om_real_current = Om_real
                            Om_current = Om

            # Skip segment if no valid combination found
            if alpha_current is None:
                print(f'No valid channel combination found for segment {seg_i + 1}. Skipping...')
                continue

            # Retrieve best-combination variables
            alpha = alpha_current
            fbeats = fbeats_current
            mbeats = mbeats_current
            x0_real = x0_real_current
            x0 = x0_current
            Om_real = Om_real_current
            Om = Om_current
            I2_orig_real = I2_orig_real_current
            I2_orig = I2_orig_current

            # =================== Second-Order Refinement — Mutual Subtraction =======
            # With the best channel pair and orientation from Step 8:
            #   1. Re-subtract fECG (Of) from the mix → cleaner mECG input
            #   2. Re-extract maternal beats on cleaner signal (lower λ = tighter DP)
            #   3. Re-subtract mECG → cleaner fECG residual
            #   4. Detect fetal beats by two independent methods:
            #        SAVER: de-shape STFT pipeline (fbeats_ma)
            #        Pan-Tompkins: classical energy-based QRS detector (fbeats_pan)
            #   5. Select the method returning more beats, bounded by 30% change vs. last segment
            #   6. Build final morphology-preserving fECG template via nonlocal-median shrinkage
            #   7. Fine-align beat indices to local signal peaks (align_beats_to_ecg)

            try:
                # Step 1: remove fetal template from signal to get cleaner mECG
                Of, Of_real = ECG_shrinkage0(I2_orig, I2_orig_real, fbeats,
                                             1.5, 0)
                Om_raw = x0 - Of
                Om_raw_real = x0_real - Of_real

                # Step 2: re-extract maternal beats from the fetal-cleaned signal
                _, _, mbeats, _, _, _, tfrrM, HR_ma, _, _ = \
                    mbeats_extract_real(Om_raw, Om_raw_real, fs, basicTF, advTF,
                                        cepR, P, lam_curve, lam_beat)

                # Step 3: re-estimate maternal ECG with updated beat locations
                Om, Om_real = ECG_shrinkage0(Om_raw, Om_raw_real, mbeats,
                                             1.5, 0)
                I2_orig = x0 - Om
                I2_orig_real = x0_real - Om_real

                # Tighten DP smoothness penalties for the refinement pass
                lam_curve = 5
                lam_beat = 5

                # Step 5a: fetal beats by SAVER SST method (using maternal HR)
                _, _, fbeats_ma, _, _, _, _, _, _, _ = \
                    fbeats_extract_real(I2_orig, I2_orig_real, fs, basicTF, advTF,
                                        cepR, P, lam_curve, lam_beat, HR_ma)

                # Determine fECG polarity (positive or negative R-peaks) then run Pan-Tompkins
                if len(fbeats_ma) > 0 and np.mean(I2_orig_real[fbeats_ma.astype(int) - 1]) > 0:
                    Po = 1
                else:
                    Po = -1
                _, fbeats_pan, _ = pan_tompkin_revised(Po * I2_orig_real, fs, 0)

                # Pick the detector that finds more beats, subject to a 30% continuity guard
                pct_change_range = 0.3
                if len(fbeats_pan) > len(fbeats_ma):
                    fbeats_len_pct_change = abs(len(fbeats_pan) - last_fbeats_len) / last_fbeats_len
                    use_pan = (not np.isnan(last_fbeats_len) and
                               (fbeats_len_pct_change < pct_change_range or
                                (fbeats_len_pct_change > pct_change_range and
                                 last_fbeats_len < 90)))
                else:
                    use_pan = False

                if use_pan:
                    fbeats = fbeats_pan
                    print('Picked PT beats!')
                else:
                    fbeats = fbeats_ma
                    print('Picked SAVER beats!')

                last_fbeats_len = len(fbeats)

                # Final fECG template: nonlocal-median shrinkage groups beats by similar
                # RRI and averages them, producing a morphology-preserving fECG waveform
                _, Of_real = ECG_shrinkage_median(I2_orig, I2_orig_real, fbeats,
                                                  num_nonlocal, 1.5, 0)

                # ============================ POSTPROCESSING ============================
                # Fine-align detected beats to nearest local maximum (mECG) / minimum (fECG)
                # within a ±5-sample and ±2-sample window respectively
                mbeats, m_ctr = align_beats_to_ecg(mbeats, Om_real, 5)
                print(f'Realigned {m_ctr}/{len(mbeats)} mbeats to extracted mECG R peaks...')

                fbeats, f_ctr = align_beats_to_ecg(fbeats, Of_real, 2)
                print(f'Realigned {f_ctr}/{len(fbeats)} fbeats to extracted fECG R peaks...')

            except Exception as e:
                print(f'Decomposition failed due to high signal noise. Skipping segment...')
                Of_real = np.zeros(len(Om_real))
                fbeats = np.zeros(1, dtype=int)

            # =================== Trim Padding & Filter Beats ===
            # Remove the 1-second padding from all waveforms; adjust beat indices to the
            # trimmed coordinate frame.

            start_ind_final = 1 + (start_ind - start_ind_padded)
            end_ind_final = len(x0_real) - (end_ind_padded - end_ind)

            x0_real = x0_real[start_ind_final - 1:end_ind_final]
            Om_real = Om_real[start_ind_final - 1:end_ind_final]
            Of_real = Of_real[start_ind_final - 1:end_ind_final]

            og_ecg_seg = []
            for ch in range(1, NUM_CHANNELS + 1):
                og_ecg_seg.append(og_ecg[ch][start_ind - 1:end_ind])

            # Remove beats that fall within the trimmed padding regions and shift
            # remaining indices to be relative to the unpadded segment start
            pad_offset = start_ind - start_ind_padded

            mask_m = (mbeats >= start_ind_final) & (mbeats <= end_ind_final)
            mbeats = mbeats[mask_m] - pad_offset

            if len(fbeats) > 0:
                mask_f = (fbeats >= start_ind_final) & (fbeats <= end_ind_final)
                fbeats = fbeats[mask_f] - pad_offset

            # =================== Store Segment Results ===========================

            final_file_names.append(file_name)
            final_start_times.append((start_ind - 1) / fs)
            og_ecg_obj = np.empty((1,), dtype=object)
            og_ecg_obj[0] = og_ecg_seg
            final_og_ECG.append(og_ecg_obj)
            final_SQI.append(bsqi_index_max)
            final_chs_used.append(np.array([[chs_used[0], chs_used[1]]], dtype=int))
            final_Om.append(Om_real)
            final_Of.append(Of_real)
            final_aECG.append(x0_real)
            final_mbeats.append(mbeats)
            final_fbeats.append(fbeats)
            final_proc_time.append(time.time() - start_clock + time_offset)

            # Persist results after each segment so a crash does not lose progress.
            parsave_output_file(
                output_file_path,
                final_file_names, final_start_times, final_og_ECG,
                final_SQI, final_chs_used, final_Om,
                final_Of, final_aECG, final_mbeats,
                final_fbeats, final_proc_time)

        elapsed = time.time() - start_clock
        print(f'Total elapsed: {elapsed:.1f} seconds')

if __name__ == '__main__':
    main()
