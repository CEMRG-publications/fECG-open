% PURPOSE:  Full Pan-Tompkins QRS detector with adaptive thresholding and T-wave
%           discrimination. Used as an independent cross-check for fetal beat detection
%           alongside the SAVER de-shape STFT pipeline.
% INPUTS:   ecg — N×1 double, ECG signal (polarity-corrected by caller)
%           fs  — scalar, sampling rate (Hz), nominally 1000
%           gr  — 0/1, plot flag (always 0 in this codebase)
% OUTPUTS:  qrs_amp_raw — 1×K double, amplitudes of detected R-peaks
%           qrs_i_raw   — 1×K integer, sample indices of detected R-peaks
%           delay       — scalar, filter group delay in samples (always 0 here)
% METHOD:   findQRSpeaks → adaptive threshold (0.75·mean_8) → search-back at 1.5×mean_RR →
%           inter-beat spacing guard (≥ 0.70·mean_RR). NOTE: the Pan-Tompkins T-wave
%           discrimination rule (accept candidate within 360 ms only if its peak slope
%           exceeds 50% of the prior QRS slope) is NOT implemented. The variable 'skip'
%           is reset to 0 in multiple places but is never set to 1 anywhere in this
%           function body. Reference: Pan & Tompkins (1985).
function [qrs_amp_raw, qrs_i_raw, delay] = pan_tompkin_revised(ecg, fs, gr)
    % Complete implementation of Pan-Tompkins algorithm
    %
    % Inputs
    %   ecg : raw ecg vector signal, 1d
    %   fs  : sampling frequency (e.g. 200Hz, 400Hz)
    %   gr  : flag to plot (1) or not plot (0)
    % Outputs
    %   qrs_amp_raw : amplitude of R waves
    %   qrs_i_raw   : index of R waves
    %   delay       : number of samples the signal is delayed due to filtering

    if ~isvector(ecg)
        error('ecg must be a row or column vector');
    end

    if nargin < 3
        gr = 1;
    end
    ecg = ecg(:);

    %% Initialize
    qrs_c = [];
    qrs_i = [];
    SIG_LEV = 0;
    nois_c = [];
    nois_i = [];
    delay = 0;
    skip = 0;
    not_nois = 0;
    selected_RR = [];
    m_selected_RR = 0;
    mean_RR = 0;
    qrs_i_raw = [];
    qrs_amp_raw = [];
    ser_back = 0;
    test_m = 0;
    SIGL_buf = [];
    NOISL_buf = [];
    THRS_buf = [];
    SIGL_buf1 = [];
    NOISL_buf1 = [];
    THRS_buf1 = [];

    [locs, pks] = findQRSpeaks(ecg, 0.2*fs);

    THR_SIG = max(ecg(1:2*fs))*1/3;
    sigAmpThreshold = THR_SIG;
    mean_RR = 0;
    THR_NOISE = mean(ecg(1:2*fs))*3/4;
    SIG_LEV = THR_SIG;
    NOISE_LEV = THR_NOISE;

    for i = 1:length(pks)

        y_i = pks(i);
        x_i = locs(i);

        if length(qrs_c) >= 9
            diffRR = diff(qrs_i(end-8:end));
            mean_RR = mean(diffRR);
            mean_8qrs = mean(qrs_c((end-7:end)));
            sigAmpThreshold = 0.75*mean_8qrs;
            comp = qrs_i(end) - qrs_i(end-1);
            m_selected_RR = mean_RR;
        elseif length(qrs_c) >= 1
            if length(qrs_c) >= 2
                diffRR = diff(qrs_i(1:end));
                mean_RR = mean(diffRR);
            else
                mean_RR = 0;
            end
            mean_qrs = mean(qrs_c((1:end)));
            sigAmpThreshold = 0.75*mean_qrs;
        end

        if m_selected_RR
            test_m = m_selected_RR;
        elseif mean_RR && m_selected_RR == 0
            test_m = mean_RR;
        else
            test_m = 0;
        end

        if test_m
            if (locs(i) - qrs_i(end)) >= round(1.5*test_m)
                [pks_temp, locs_temp] = max(ecg(qrs_i(end) + round(0.200*fs):locs(i) - round(0.200*fs)));
                locs_temp = qrs_i(end) + round(0.200*fs) + locs_temp - 1;

                qrs_c = [qrs_c pks_temp];
                qrs_i = [qrs_i locs_temp];

                y_i_t = pks_temp;
                x_i_t = locs_temp;

                not_nois = 1;
            else
                not_nois = 0;
            end
        end

        if pks(i) >= sigAmpThreshold
            if length(qrs_i) > 1
                if (locs(i) - qrs_i(end)) >= 0.70*mean_RR && skip == 0
                    qrs_c = [qrs_c pks(i)];
                    qrs_i = [qrs_i locs(i)];
                end
            else
                if skip == 0
                    qrs_c = [qrs_c pks(i)];
                    qrs_i = [qrs_i locs(i)];
                end
            end
            skip = 0;
            not_nois = 0;
            ser_back = 0;
        end
        qrs_i_raw = qrs_i;
        qrs_amp_raw = qrs_c;
        THRS_buf = [THRS_buf sigAmpThreshold];
    end
end
