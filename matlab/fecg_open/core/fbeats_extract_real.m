% PURPOSE:  Extract fetal ECG R-peaks from a rough fECG residual signal, suppressing
%           any residual maternal HR energy in the time-frequency plane before tracking.
% INPUTS:   I2_orig      — 1×N double, rough fECG residual (for timing)
%           I2_orig_morph— 1×N double, morphology-preserving fECG residual
%           fs           — scalar, sampling rate (Hz), nominally 1000
%           basicTF      — struct: STFT parameters (win compressed to 3/5 for fetal)
%           advTF        — struct: SST thresholds
%           cepR         — struct: cepstral parameters
%           P            — struct: P.num_s=1
%           lam_curve    — scalar ≥0, DP smoothness weight for HR ridge
%           lam_beat     — scalar ≥0, DP transition weight for beat tracker
%           HR_ma        — 1×T double, maternal HR (bins) per TF frame — used to
%                          suppress maternal residue in the fetal TFR
% OUTPUTS:  (same layout as mbeats_extract_real but for fetal beats)
%           I2_orig      — polarity-corrected fECG residual
%           I2_orig_morph— polarity-corrected morphology residual
%           fbeats_p     — 1×K integer, fetal R-peak sample indices
%           fbeats_q     — 1×K integer, fetal S-point sample indices
%           R_amp / S_amp— scalar amplitudes
%           tfrrF        — freq×time, fetal TFR after maternal suppression
%           HR_fe        — 1×T double, estimated fetal HR (bins) per frame
%           tfrtic / t   — frequency and time axes
% METHOD:   CFPH (SST, shorter window) → suppress maternal HR band in TFR →
%           CurveExt_M (DP ridge) → beat_simple → RRconstraint → peak search
function [I2_orig, I2_orig_morph, fbeats_p, fbeats_q, R_amp, S_amp, tfrrF, HR_fe, tfrtic, t] = fbeats_extract_real(I2_orig, I2_orig_morph, fs, basicTF, advTF, cepR, P, lam_curve, lam_beat, HR_ma)

    I2_orig_len = floor(length(I2_orig)/200)*200;
    I2 = resample(I2_orig, 100, fs);
    basicTF.win = round(basicTF.win*3/5);

    gg = I2; gg = gg - mean(gg);

    % apply the de-shape on the rough fECG to get the fetal HR
    [~, ~, ~, tfrrF, ~, ~, tfrtic, t] = CFPH(gg, basicTF, advTF, cepR, P);

    % suppress the possible residue of the mECG
    for ti = 1:size(tfrrF, 2)
        idx = round(HR_ma(ti)*0.94):round(HR_ma(ti)*1.06);
        if max(HR_ma >= size(tfrrF, 1))
            continue;
        else
            tfrrF(idx, ti) = tfrrF(idx, ti)/10;
        end
    end

    HR_fe = Extract_fhr(tfrrF, basicTF, lam_curve);
    HR_fe3 = interp1(200:200:length(I2_orig), HR_fe, 1:length(I2_orig), 'pchip', 'extrap');

    % apply the beat tracking to get the fetal HR
    flocsf1p = beat_simple(I2, 100, HR_fe3.*basicTF.fr, lam_beat);
    flocsf1q = beat_simple(-I2, 100, HR_fe3.*basicTF.fr, lam_beat);

    % get the fetal polarity
    if abs(median(I2(flocsf1p))) > abs(median(I2(flocsf1q)))
        Po = 1; flocsf = flocsf1p;
    else
        Po = -1; flocsf = flocsf1q;
        fprintf('\t\t*** reverse the fetal pole\n');
    end

    I2 = Po.*I2;
    I2_orig = Po.*I2_orig;
    I2_orig_morph = Po.*I2_orig_morph;

    flocsf = RRconstraint(flocsf, I2, 100, 0.25);

    I2 = I2_orig;
    flocsf = round(flocsf*10/1);
    fbeats_p = [];
    fbeats_q = [];
    SearchLen_p = round(100./mean(HR_fe3*basicTF.fr));
    SearchLen_q = 100;

    for ii = 1:length(flocsf)
        [~, idx] = max(I2(max([1 flocsf(ii)-SearchLen_p]):min([flocsf(ii)+SearchLen_p length(I2)])));
        fbeats_p(ii) = flocsf(ii) - SearchLen_p + idx - 1;
        [~, idx2] = min(I2(max([1 flocsf(ii)-SearchLen_q]):min([flocsf(ii)+SearchLen_q length(I2)])));
        fbeats_q(ii) = flocsf(ii) - SearchLen_q + idx2 - 1;
    end
    ind = find(fbeats_p > 0 & fbeats_p < I2_orig_len & fbeats_q > 0 & fbeats_q < I2_orig_len);
    fbeats_p = fbeats_p(ind);
    fbeats_q = fbeats_q(ind);
    R_amp = median(I2(fbeats_p));
    S_amp = median(I2(fbeats_q));

    I2_orig = Po.*I2_orig;
    I2_orig_morph = Po.*I2_orig_morph;

end
