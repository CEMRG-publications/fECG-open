clc
clear
close all

input_folder = '../data/inputs';
output_folder = '../data/outputs/Matlab';

if ~isfolder(input_folder)
    error('Non-existing input folder: %s', input_folder)
end

if ~isfolder(output_folder)
    mkdir(output_folder)
end

% ======================================================================= %

addpath('./fecg_open/core')
addpath('./fecg_open/core/peak_detector')
addpath('./fecg_open/utils')

%% STEP 1: Recording & Algorithm Parameters
% All numerical constants below are fixed for this cohort.
% SQI options govern how beat-detection quality is measured (see detect_bsqi).
% TF (time-frequency) options govern the de-shape STFT used to track heart rate.
% Morphology options govern median-filter window sizes for baseline removal.
NUM_CHANNELS = 4;
fs = 1000; % Sampling rate (Hz) — target after resampling all channels

%% ==================== SQI parameters ===============================
% bSQI = beat-by-beat Signal Quality Index. Scores 0–1 how well two
% independent QRS detectors agree. A score > SQI_THR (0.8) is "good".
opt = struct(...
    'SIZE_WIND', 4,...    % window size (s) for each bSQI evaluation window
    'LG_MED', 0,...       % median smoothing across adjacent SQI windows (0 = off)
    'REG_WIN', 1,...      % how often (s) SQI is re-evaluated
    'THR', 0.150,...      % tolerance (s) for calling two detected beats a match
    'SQI_THR', 0.8,...    % minimum SQI to consider a signal "good quality"
    'JQRS_THRESH', 0.3,...  % jqrs energy threshold (relative to signal max)
    'JQRS_REFRAC', 0.25,... % jqrs refractory period (s) — minimum inter-beat gap
    'JQRS_INTWIN_SZ', 7,... % jqrs integration window size (samples at fs)
    'JQRS_WINDOW', 15);     % jqrs processing sub-window size (s)

%% ============================== De-shape STFT (TF analysis) parameters ===
% These control the Synchrosqueezing Transform used to estimate instantaneous
% heart rate as a ridge in the time-frequency plane.
basicTF.win = 1000;  % STFT window length (samples at 100 Hz internal rate = 10 s)
basicTF.hop = 20;    % STFT hop size (samples) — controls time resolution
basicTF.fs = 100;    % internal resampled rate (Hz) used inside the TF pipeline
basicTF.fr = 0.02;   % frequency resolution (cycles per sample at basicTF.fs)
advTF.ths = 1E-6;    % synchrosqueezing threshold — suppress near-zero TFR entries
advTF.HighFreq = 10/100; % upper heart rate limit (fraction of basicTF.fs = 10 Hz → 600 bpm)
advTF.LowFreq = 0.5/100; % lower heart rate limit (0.5 Hz → 30 bpm)
cepR.g = 0.3;  % cepstral exponent — controls how sharply harmonic peaks are enhanced
cepR.Tc = 0;   % cepstral threshold — zero values below this level
P.num_s = 1;   % number of synchrosqueeze harmonics (1 = fundamental only)
alphaN = 7;    % half-width of the α grid; generates 2*alphaN=14 channel-mix orientations

%% ============================== Morphology / shrinkage parameters ========
num_med = 51;       % short median filter length (samples) for detrending without morphology
num_med_l = 301;    % long median filter length — suppresses low-frequency drift
num_med_s = 101;    % short median filter for morphology-preserving detrend
num_nonlocal = 10;  % number of nearest-RRI neighbours used in nonlocal median shrinkage

%% ============================ Input / Output Paths =======================
DB_files = dir(strcat(input_folder, '/*.ch1')); % collect all channel-1 files for this subject

%% ============================== Precomputed constants =======================

% Filters for channels 1-3 (fs_og = 300 Hz; Nyquist = 150 Hz)
[b_lp300, a_lp300] = butter(5, 120 / 150, 'low');
[b_hp300, a_hp300] = butter(5, 0.5 / 150, 'high');
[b_bs300, a_bs300] = butter(5, [48 52] / 150, 'stop');

% Filters for channel 4 (fs_og = 900 Hz; Nyquist = 450 Hz)
[b_lp900, a_lp900] = butter(5, 120 / 450, 'low');
[b_hp900, a_hp900] = butter(5, 0.5 / 450, 'high');
[b_bs900, a_bs900] = butter(5, [48 52] / 450, 'stop');

%% ============================== Outer File Loop ==========================
% Iterates over each recording file (.ch1 through .ch4) found in input_folder.
% Each file corresponds to one continuous multi-channel abdominal recording.
for file_i = 1:length(DB_files)

    tic

    %% STEP 2: Per-File Initialization — Output Containers
    % Pre-allocate cell arrays that will hold one row per 60-second segment.
    % Each entry is appended incrementally so the file can be resumed after a crash.

    % Instantiate our final outputs
    final_file_names = {};      % File name
    final_start_times = {};     % Start time of the segment (seconds)
    final_og_ECG = {};          % Original interpolated ECG (no preprocessing)
    final_SQI = {};             % Best bSQI achieved
    final_chs_used = {};        % Pair of channels used to achieve above bSQI
    final_Om = {};              % Extracted separated maternal ECG signal
    final_Of = {};              % Extracted separated fetal ECG signal
    final_aECG = {};            % Decomposed combined ECG signal
    final_mbeats = {};          % Indices of maternal heartbeats / R peaks
    final_fbeats = {};          % Indices of fetal heartbeats / R peaks
    final_proc_time = {};       % Current duration of processing

    og_ecg = cell(1, NUM_CHANNELS);      % collection of original signals
    x_up_c = cell(1, NUM_CHANNELS);      % collection of preprocessed signals without morphology
    x_up_c_real = cell(1, NUM_CHANNELS); % collection of preprocessed signals with morphology

    % Get file name and print it out
    file = DB_files(file_i);
    file_name = split(file.name, '.ch1'); file_name = file_name{1};

    fprintf('Loading [%d/%d]: %s...\n', file_i, length(DB_files), file_name);

    % Instantiate output file — use matfile() for incremental on-disk writes
    % so partial results survive a MATLAB crash.
    output_file_path = strcat(output_folder, '/', strcat(file_name, '.mat'));
    segment_offset = 0;
    time_offset = 0;
    if isfile(output_file_path)
        fprintf('Output file for %s already exists...\n', file_name);
        output_file = matfile(output_file_path, 'Writable', true);
        segment_offset = length(output_file.final_file_names);
        fprintf('Continuing decomposition after segment %d...\n', segment_offset);

        try
            time_offset = output_file.final_proc_time(end, 1);
            time_offset = time_offset{1};
            last_fbeats_len = output_file.final_fbeats(end, 1);
            last_fbeats_len = length(last_fbeats_len{1});
        catch
            time_offset = 0;
            last_fbeats_len = NaN;
        end

    else
        % Create the output .mat file with empty containers so matfile() can
        % append to it row-by-row without loading the whole file into memory.
        parsave_output_file(...
            output_file_path,...
            final_file_names,...
            final_start_times,...
            final_og_ECG,...
            final_SQI,...
            final_chs_used,...
            final_Om,...
            final_Of,...
            final_aECG,...
            final_mbeats,...
            final_fbeats,...
            final_proc_time)

        output_file = matfile(output_file_path, 'Writable', true);
    end

    %% STEP 3: Load, Filter & Detrend All Channels
    % Each .chN file is read as raw 32-bit integers, then:
    %   1. Band-pass filtered to keep only the ECG band
    %   2. Scaled to millivolts (÷1000)
    %   3. Resampled to a common 1000 Hz grid via PCHIP interpolation
    %   4. Detrended in two versions:
    %      x_up_c      — aggressive (good for R-peak detection)
    %      x_up_c_real — mild morphology-preserving (good for waveform shape)
    for ch_i = 1:NUM_CHANNELS
        % Generate the channel file name to read from
        channel_file_name = strcat(file_name, '.ch', num2str(ch_i));
        channel_file_path = fullfile(input_folder, channel_file_name);

        % Read the channel signal
        channel_signal = MyReadDataq_32(channel_file_path);
        channel_signal = channel_signal - mean(channel_signal);
        channel_signal = channel_signal'; % Transpose to one row

        % ============================== PREPROCESSING ==============================

        % Channels 1-3 were recorded at 300 Hz; channel 4 at 900 Hz.
        % All will be resampled to fs=1000 Hz below.
        if ch_i < 4 % Channels 1-3 are 300 Hz
            fs_og = 300;
            [b_low, a_low] = deal(b_lp300, a_lp300);
            [b_high, a_high] = deal(b_hp300, a_hp300);
            [b_notch, a_notch] = deal(b_bs300, a_bs300);
        else % Channel 4 is 900 Hz
            fs_og = 900;
            [b_low, a_low] = deal(b_lp900, a_lp900);
            [b_high, a_high] = deal(b_hp900, a_hp900);
            [b_notch, a_notch] = deal(b_bs900, a_bs900);
        end

        fprintf('Preprocessing [%d/%d]: %s...\n', ch_i, NUM_CHANNELS, channel_file_name);

        % Three-stage zero-phase Butterworth filter cascade (filtfilt = no phase shift):
        %   Lowpass  120 Hz  — removes high-frequency EMG and electronic noise
        %   Highpass 0.5 Hz  — removes slow baseline wander from breathing/movement
        %   Notch  48-52 Hz  — suppresses 50 Hz powerline interference (±2 Hz guard)
        % 5th-order Butterworth gives a steep roll-off with minimal passband ripple.
        channel_signal = filtfilt(b_low, a_low, channel_signal);
        channel_signal = filtfilt(b_high, a_high, channel_signal);
        channel_signal = filtfilt(b_notch, a_notch, channel_signal);

        % Scale down from raw integer units to a millivolt-scale range.
        channel_signal = channel_signal/1000;

        % Resample to a common 1000 Hz grid using piecewise cubic Hermite
        % interpolation (pchip) — preserves peak amplitudes better than linear.
        channel_signal_indices = linspace(0, length(channel_signal)/fs_og, length(channel_signal));
        channel_signal_indices_new = linspace(0, channel_signal_indices(end), channel_signal_indices(end)*fs);
        channel_signal = interp1(channel_signal_indices, channel_signal, channel_signal_indices_new, 'pchip');

        % Store original signal in memory
        og_ecg{ch_i} = channel_signal;

        % Baseline removal via cascaded median filter + loess smoothing.
        % Two versions: x0 uses an aggressively short window (51 samples ≈ 50 ms)
        % that strips waveform morphology but gives clean R-peak timing; x0_real
        % uses a gentler 101-sample window that preserves beat shapes.
        x0 = ECG_detrend(channel_signal, num_med, num_med_l, 0);      % no morphology
        x0_real = ECG_detrend(channel_signal, num_med_s, num_med_l, 1); % preserve morphology

        % Store in memory
        x_up_c{ch_i} = x0;
        x_up_c_real{ch_i} = x0_real;
    end

    %% STEP 4: Segment the Recording into 60-Second Windows
    % The full recording is split into non-overlapping 60-second segments
    % (60000 samples at 1000 Hz). Processing is performed independently per
    % segment. A 1-second padding is added at each boundary to avoid edge
    % artefacts; padding is trimmed after separation.
    full_length = length(x_up_c{1});
    full_length = full_length - mod(full_length, fs); % round down to whole seconds
    SEGMENT_TIME_RANGES = 0 : 60*fs : full_length;
    % Add any remaining < 1 minute reading as a segment at the end
    if full_length - SEGMENT_TIME_RANGES(end) > 0
        SEGMENT_TIME_RANGES(length(SEGMENT_TIME_RANGES)+1) = full_length;
    end
    NUM_SEGMENTS = length(SEGMENT_TIME_RANGES) - 1;

    %% STEP 5: Per-Segment ECG Separation Loop
    % For each 60-second segment, exhaustively search all channel pairs and
    % signal orientations to find the combination that best separates maternal
    % and fetal ECG. Separation quality is measured by the bSQI score.
    for seg_i = 1+segment_offset : NUM_SEGMENTS
        % DP smoothness weights — loosened here, tightened in the refinement pass
        lam_curve = 50; % TF ridge extraction penalty (higher → smoother HR curve)
        lam_beat = 50;  % beat-tracking transition penalty (higher → more regular rhythm)

        start_ind = SEGMENT_TIME_RANGES(seg_i) + 1;
        end_ind = SEGMENT_TIME_RANGES(seg_i + 1);

        % Add 1 second to the beginning and end to avoid filter edge artefacts
        start_ind_padded = start_ind - (fs * 1);
        end_ind_padded = end_ind + (fs * 1);

        % Ensure start and end indices aren't out of bound
        start_ind_padded = max(start_ind_padded, 1);
        end_ind_padded = min(end_ind_padded, full_length);

        fprintf('Decomposing [%d/%d]: %s [%d:%d]...\n', seg_i, NUM_SEGMENTS, file_name, start_ind_padded, end_ind_padded);

        %% STEP 6: Exhaustive Channel-Pair & Orientation Search
        % Try all C(NUM_CHANNELS,2) electrode pairs. For each pair, generate
        % 2*alphaN unit-norm linear combinations:
        %          x0 = √(1-α²)·sig1 + α·sig2, where α ∈ [-1,1].
        % The best combination (highest bSQI on the fetal signal) is retained.
        bsqi_index_max = -inf;
        chs_used = [];
        for ch_i = 1:(NUM_CHANNELS - 1)
            for ch_i2 = (ch_i + 1):NUM_CHANNELS
                % Read data
                sig1 = x_up_c{ch_i}(start_ind_padded:end_ind_padded);
                sig2 = x_up_c{ch_i2}(start_ind_padded:end_ind_padded);
                sig1_real = x_up_c_real{ch_i}(start_ind_padded:end_ind_padded);
                sig2_real = x_up_c_real{ch_i2}(start_ind_padded:end_ind_padded);

                %% STEP 7: α-Weighted Combination & Maternal Beat Extraction
                % 2*alphaN orientations are tested. For each, the de-shape STFT pipeline
                % (mbeats_extract_real) estimates maternal HR as a TF ridge, then uses
                % dynamic-programming beat tracking to locate R-peaks. HRV (second-
                % difference RMSSD) is computed; the 5 most regular orientations are
                % passed to the multi-channel fusion step.
                alpha_ind = ((1:alphaN * 2) - alphaN) ./ alphaN; % 2*alphaN α values in [-1,1]
                alpha_sig_c = cell(1, length(alpha_ind));   % mixed signals per orientation
                alpha_mbeats_c = cell(1, length(alpha_ind)); % detected beats per orientation
                alpha_hrv_c = zeros(1, length(alpha_ind));   % HRV score per orientation

                for i = 1:length(alpha_ind)
                    alpha = alpha_ind(i);

                    % Unit-norm linear combination of the two channels
                    x0 = sqrt(1 - alpha ^ 2) .* sig1 + alpha .* sig2;
                    x0_real = sqrt(1 - alpha ^ 2) .* sig1_real + alpha .* sig2_real;

                    % Get maternal R peaks
                    [x0, ~, mbeats, ~, ~, ~, ~, ~, ~, ~] = mbeats_extract_real(x0, x0_real, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat);
                    alpha_sig_c{i} = x0;
                    alpha_mbeats_c{i} = mbeats;
                    % RMSSD-like HRV: low value → regular rhythm → good maternal ECG orientation
                    hrv = abs(diff(diff(mbeats)));
                    hrv = sqrt(sum(hrv .^ 2) / length(hrv));
                    alpha_hrv_c(i) = hrv;
                end

                % Select the 5 most regular (lowest HRV) orientations for fusion
                [~, index] = sort(alpha_hrv_c);
                index = index(1:5);
                asig_c = cell(1, length(index));
                ambeats_c = cell(1, length(index));
                for jj = 1:length(index)
                    asig_c{jj} = alpha_sig_c{index(jj)};
                    ambeats_c{jj} = alpha_mbeats_c{index(jj)};
                end

                % Multi-channel beat fusion: combines R-peak lists from the 5 best
                % orientations using template correlation and RRI regularity scoring,
                % then clusters nearby detections to produce consensus beat times.
                mbeats_fus = fusion_mbeats(ambeats_c, asig_c, fs);

                if length(mbeats_fus) > 20
                    % Convert fused beat times to an interpolated HR vector (bpm)
                    % sampled every 200 ms — used to suppress maternal HR in the fetal TFR
                    RRI = diff(mbeats_fus); RRI = [RRI(1), RRI];
                    HR_ma_fus = interp1(mbeats_fus, 60 * fs ./ RRI, 1:200:length(x0), 'nearest', 'extrap');
                    HR_ma_fus = round(HR_ma_fus);
                    HR_ma_fus(HR_ma_fus < 0) = 10; % clamp negative artefacts
                else
                    % Fallback if fusion fails: use beats from the single best orientation
                    disp('bad fusion result, use channel 1 R peaks');
                    [sig1, sig1_real, mbeats_p1, ~, ~, ~, ~, HR_ma_fus, ~, ~] = mbeats_extract_real(sig1, sig1_real, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat);
                    mbeats_fus = mbeats_p1;
                end

                %% STEP 8: Rough Fetal Signal Estimation & bSQI Scoring
                % For each of the 2*alphaN α-mixes:
                %   1. Snap fused maternal beats to local signal peaks (mbeats_modify)
                %   2. Subtract the mECG template (SVD optimal shrinkage, ECG_shrinkage0)
                %   3. Detect fetal R-peaks in the residual (fbeats_extract_real)
                %   4. Score fetal beat quality (bSQI: jqrs vs. gqrs agreement)
                % Retain the α + channel pair that yields the highest bSQI.
                for i = 1:length(alpha_ind)
                    alpha = alpha_ind(i);
                    x0 = sqrt(1 - alpha ^ 2) .* sig1 + alpha .* sig2;
                    x0_real = sqrt(1 - alpha ^ 2) .* sig1_real + alpha .* sig2_real;

                    % Snap fused maternal beats to nearest local peak in x0
                    [mbeats, ~, ~, ~, ~] = mbeats_modify(x0, mbeats_fus);
                    try
                        % SVD-based template subtraction: builds a beat matrix, applies
                        % optimal shrinkage (op norm) to isolate the periodic mECG component
                        [Om, Om_real] = ECG_shrinkage0(x0, x0_real, mbeats, 1.5, 0);
                    catch
                        fprintf('Decomposition failed due to high signal noise. Skipping segment...\n');
                        Om = zeros(1, length(x0));
                        Om_real = zeros(1, length(x0_real));
                    end

                    I2_orig_real = x0_real - Om_real; % rough fECG residual (morph-preserving)
                    I2_orig = x0 - Om;                % rough fECG residual (beat-detection)

                    % Detect fetal R-peaks using the de-shape STFT pipeline,
                    % with maternal HR suppressed in the TFR to avoid confusion
                    try
                        [~, ~, fbeats, ~, ~, ~, ~, ~, ~, ~] = fbeats_extract_real(I2_orig, I2_orig_real, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat, HR_ma_fus);
                    catch
                        continue
                    end
                    if isempty(fbeats); continue; end

                    % bSQI: compares fetal R-peaks (acting as gqrs reference) against
                    % jqrs detections on the fECG residual. Median across windows.
                    [~, sqi_f] = detect_bsqi(I2_orig', {'ECG'}, fs, opt, fbeats);
                    bsqi_index = median(sqi_f{1});

                    % Only save this combination of channels if it has the highest bSQI so far
                    if (bsqi_index > bsqi_index_max)
                        fprintf('Found channels: [%d, %d] w/ higher bSQI: %6.3f\n', ch_i, ch_i2, bsqi_index);
                        alpha_current = alpha;
                        bsqi_index_max = bsqi_index;
                        chs_used = [ch_i, ch_i2];
                        mbeats_current = mbeats;
                        fbeats_current = fbeats;
                        x0_current = x0;
                        x0_real_current = x0_real;
                        I2_orig_current = I2_orig;
                        I2_orig_real_current = I2_orig_real;
                        Om_real_current = Om_real;
                        Om_current = Om;
                    end
                end

            end
        end

        % Retrieve the best-scoring combination
        alpha = alpha_current;
        fbeats = fbeats_current;
        mbeats = mbeats_current;
        x0_real = x0_real_current;
        x0 = x0_current;
        Om_real = Om_real_current;
        Om = Om_current;
        I2_orig_real = I2_orig_real_current;
        I2_orig = I2_orig_current;

        %% STEP 9: Second-Order Refinement — Mutual Subtraction
        % With the best channel pair and orientation from Step 8:
        %   1. Re-subtract fECG (Of) from the mix → cleaner mECG input
        %   2. Re-extract maternal beats on cleaner signal (lower λ = tighter DP)
        %   3. Re-subtract mECG → cleaner fECG residual
        %   4. Detect fetal beats by two independent methods:
        %        SAVER: de-shape STFT pipeline (fbeats_ma)
        %        Pan-Tompkins: classical energy-based QRS detector (fbeats_pan)
        %   5. Select the method returning more beats, bounded by 30% change vs. last segment
        %   6. Build final morphology-preserving fECG template via nonlocal-median shrinkage
        %   7. Fine-align beat indices to local signal peaks (align_beats_to_ecg)
        try
            [Of, Of_real] = ECG_shrinkage0(I2_orig, I2_orig_real, fbeats, 1.5, 0);
            Om_raw = x0 - Of; Om_raw_real = x0_real - Of_real;
            [~, ~, mbeats, ~, ~, ~, ~, HR_ma, ~, ~] = mbeats_extract_real(Om_raw, Om_raw_real, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat);
            [Om, Om_real] = ECG_shrinkage0(Om_raw, Om_raw_real, mbeats, 1.5, 0);
            I2_orig = x0 - Om; I2_orig_real = x0_real - Om_real;

            % Tighten DP smoothness penalties for the refinement pass
            lam_curve = 5;
            lam_beat = 5;

            % Calculate fbeats in 2 different ways
            [~, ~, fbeats_ma, ~, ~, ~, ~, ~, ~, ~] = fbeats_extract_real(I2_orig, I2_orig_real, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat, HR_ma);

            % Determine fECG polarity (positive or negative R-peaks) then run Pan-Tompkins
            if mean(I2_orig_real(fbeats_ma)) > 0; Po = 1; else; Po = -1; end
            [~, fbeats_pan, ~] = pan_tompkin_revised(Po*I2_orig_real, fs, 0);

            % Pick the detector that finds more beats, subject to a 30% continuity guard:
            % if fbeats_pan returns > fbeats_ma AND the count is within 30% of the
            % previous segment's count, prefer Pan-Tompkins (or if count < 90 beats/min)
            pct_change_range = 0.3;
            if length(fbeats_pan) > length(fbeats_ma)
                fbeats_len_pct_change = abs(length(fbeats_pan) - last_fbeats_len) / last_fbeats_len;
                if ~isnan(last_fbeats_len) && fbeats_len_pct_change < pct_change_range
                    fbeats = fbeats_pan;
                    fprintf('Picked PT beats!\n')
                elseif ~isnan(last_fbeats_len) && fbeats_len_pct_change > pct_change_range && last_fbeats_len < 90
                    fbeats = fbeats_pan;
                    fprintf('Picked PT beats!\n')
                else
                    fbeats = fbeats_ma;
                    fprintf('Picked SAVER beats!\n')
                end
            else
                fbeats = fbeats_ma;
                fprintf('Picked SAVER beats!\n')
            end

            last_fbeats_len = length(fbeats);

            % Final fECG template: nonlocal-median shrinkage groups beats by similar
            % RRI and averages them, producing a morphology-preserving fECG waveform
            [~, Of_real] = ECG_shrinkage_median(I2_orig, I2_orig_real, fbeats, num_nonlocal, 1.5, 0);

            % ============================ POSTPROCESSING ============================
            % Fine-align detected beats to nearest local maximum (mECG) / minimum (fECG)
            % within a ±5-sample and ±2-sample window respectively
            [mbeats, m_ctr] = align_beats_to_ecg(mbeats, Om_real, 5);
            fprintf('Realigned %d/%d mbeats to extracted mECG R peaks...\n', m_ctr, length(mbeats));
            [fbeats, f_ctr] = align_beats_to_ecg(fbeats, Of_real, 2);
            fprintf('Realigned %d/%d fbeats to extracted fECG R peaks...\n', f_ctr, length(fbeats));
        catch
            fprintf('Decomposition failed due to high signal noise. Skipping segment...\n');
            Of_real = zeros(1, length(Om_real));
            fbeats = zeros(1, 1);
        end

        %% STEP 10: Trim Padding & Filter Beat Indices
        % Remove the 1-second padding that was added in Step 5 from all waveforms.
        % Adjust beat sample indices to the trimmed coordinate frame.

        % Trim previously added padding from start and end of ECGs
        start_ind_final = 1 + (start_ind - start_ind_padded);
        end_ind_final = length(x0_real) - (end_ind_padded - end_ind);
        x0_real = x0_real(start_ind_final:end_ind_final);
        Om_real = Om_real(start_ind_final:end_ind_final);
        Of_real = Of_real(start_ind_final:end_ind_final);

        % Get the corresponding segment from original ECG signals
        og_ecg_seg = og_ecg;
        for og_ch_i = 1:length(og_ecg_seg)
            og_ecg_seg{og_ch_i} = og_ecg_seg{og_ch_i}(start_ind:end_ind);
        end

        % Remove beats that fall within the trimmed padding regions and shift
        % remaining indices to be relative to the unpadded segment start
        mbeats = mbeats(mbeats >= start_ind_final); mbeats = mbeats(mbeats <= end_ind_final); mbeats = mbeats - (start_ind - start_ind_padded);
        fbeats = fbeats(fbeats >= start_ind_final); fbeats = fbeats(fbeats <= end_ind_final); fbeats = fbeats - (start_ind - start_ind_padded);

        %% STEP 11: Persist Segment Results to Disk
        % Append this segment's outputs to the MAT-file using matfile() for
        % incremental on-disk writes. Processing time is accumulated so runs
        % can be resumed and timing reported correctly across sessions.
        output_file.final_file_names(seg_i, 1) = {file_name};
        output_file.final_start_times(seg_i, 1) = {(start_ind - 1) / fs};
        output_file.final_og_ECG(seg_i, 1) = {og_ecg_seg};
        output_file.final_SQI(seg_i, 1) = {bsqi_index_max};
        output_file.final_chs_used(seg_i, 1) = {chs_used};
        output_file.final_Om(seg_i, 1) = {Om_real};
        output_file.final_Of(seg_i, 1) = {Of_real};
        output_file.final_aECG(seg_i, 1) = {x0_real};
        output_file.final_mbeats(seg_i, 1) = {mbeats};
        output_file.final_fbeats(seg_i, 1) = {fbeats};
        output_file.final_proc_time(seg_i, 1) = {toc + time_offset};
    end

     toc
end
