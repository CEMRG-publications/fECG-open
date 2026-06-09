% PURPOSE:  Extract maternal ECG R-peaks from an abdominal mixture signal using
%           a time-frequency ridge-following and dynamic-programming beat tracker.
% INPUTS:   x0        — 1×N double, de-trended mixed ECG (for beat timing)
%           x0_real   — 1×N double, morphology-preserving de-trended mixed ECG
%           fs        — scalar, signal sampling rate (Hz), nominally 1000
%           basicTF   — struct: STFT parameters (win, hop, fs=100, fr=0.02)
%           advTF     — struct: SST thresholds (ths, HighFreq, LowFreq)
%           cepR      — struct: cepstral parameters (g=0.3, Tc=0)
%           P         — struct: P.num_s=1
%           lam_curve — scalar ≥0, DP smoothness weight for HR ridge extraction
%           lam_beat  — scalar ≥0, DP transition weight for beat tracker
% OUTPUTS:  x0        — 1×N double, polarity-corrected version of x0
%           x0_real   — 1×N double, polarity-corrected morphology version
%           mbeats_p  — 1×K integer, sample indices of detected R-peaks (at fs)
%           mbeats_q  — 1×K integer, sample indices of detected S-points (at fs)
%           R_amp     — scalar, median R-peak amplitude
%           S_amp     — scalar, median S-point amplitude
%           tfrrM     — freq×time double, cepstrum-weighted TFR (for HR curve)
%           HR_ma     — 1×T double, estimated maternal heart rate (bins) per TF frame
%           tfrtic    — freq×1 double, TFR frequency axis
%           t         — 1×T double, TFR time axis
% METHOD:   Log-transform → CFPH (SST) → CurveExt_M (DP ridge) → beat_simple
%           (DP beat tracker) → RRconstraint (refractory filter) → peak search
function [x0, x0_real, mbeats_p, mbeats_q, R_amp, S_amp, tfrrM, HR_ma, tfrtic, t] = mbeats_extract_real(x0, x0_real, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat)

    segment_size = fs / 5;
    x0_len = floor(length(x0)/segment_size)*segment_size;
    x0 = x0(1:x0_len);
    x1 = resample(x0, 100, fs);

    gg = log(1+abs(x1)); gg = gg - mean(gg);

    [~, ~, ~, tfrrM, ~, ~, tfrtic, t] = CFPH(gg, basicTF, advTF, cepR, P);

    HR_ma = Extract_mhr(tfrrM, basicTF, lam_curve);
    HR_ma2 = interp1(segment_size:segment_size:x0_len, HR_ma, 1:x0_len, 'pchip', 'extrap');

    % Get maternal R peaks by DP
    mlocsf1p = beat_simple(x1, 100, HR_ma2.*basicTF.fr, lam_beat);
    mlocsf1q = beat_simple(-x1, 100, HR_ma2.*basicTF.fr, lam_beat);

    if abs(median(x1(mlocsf1p))) > abs(median(x1(mlocsf1q)))
        Po = 1; mlocsf1 = mlocsf1p;
    else
        Po = -1; mlocsf1 = mlocsf1q;
    end
    x1 = Po.*x1;
    x0 = Po.*x0;
    x0_real = Po.*x0_real;

    mlocsf1 = RRconstraint(mlocsf1, x1, 100, 0.25);

    % Get maternal ECG waveforms from nonlocal median
    x_up = x0;
    mlocsf = round(mlocsf1*10);
    mbeats_p = [];
    mbeats_q = [];
    SearchLen_p = round(48./mean(HR_ma2*basicTF.fr));
    SearchLen_q = 48;
    for ii = 1:length(mlocsf)
        [~, idx] = max(x_up(max([mlocsf(ii)-SearchLen_p 1]):min([mlocsf(ii)+SearchLen_p length(x_up)])));
        mbeats_p(ii) = mlocsf(ii) - SearchLen_p + idx - 1;
        [~, idx2] = min(x_up(max([mlocsf(ii)-SearchLen_q 1]):min([mlocsf(ii)+SearchLen_q length(x_up)])));
        mbeats_q(ii) = mlocsf(ii) - SearchLen_q + idx2 - 1;
    end
    ind = find(mbeats_p > 0 & mbeats_p < x0_len & mbeats_q > 0 & mbeats_q < x0_len);
    mbeats_p = mbeats_p(ind);
    mbeats_q = mbeats_q(ind);
    R_amp = median(x_up(mbeats_p));
    S_amp = median(x_up(mbeats_q));

    x0 = Po.*x0;
    x0_real = Po.*x0_real;

end
