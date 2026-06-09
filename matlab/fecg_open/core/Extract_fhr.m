% NOTE: Python equivalent is named extract_fhr (lowercase) in fecg_open/core/extract_fhr.py (PEP 8 rename).
% PURPOSE:  Extract the fetal heart-rate curve from a synchrosqueezed TFR by
%           masking to the fetal HR band (1.0–3.2 Hz at basicTF.fs=100 Hz,
%           i.e. 60–192 bpm) and applying dynamic-programming ridge extraction.
% INPUTS:   rtfr_post — freq×time double, synchrosqueezed TFR (after maternal suppression)
%           basicTF   — struct: basicTF.fr is the frequency resolution
%           dtw       — scalar, DP smoothness weight (= lam_curve)
% OUTPUTS:  HR — 1×time double, estimated fetal HR as frequency-bin index per frame
% METHOD:   Zero TFR outside [1/fr, 3.2/fr] bin range → CurveExt_M; offset by lower bound.
function HR = Extract_fhr(rtfr_post, basicTF, dtw)

    rtfr_post(1:round(0.4/basicTF.fr), :) = 0;
    rtfr_post(round(3.2/basicTF.fr):end, :) = 0;

    HR = CurveExt_M(rtfr_post(round(1/basicTF.fr)+1:round(3.2/basicTF.fr), :)', dtw);
    HR = HR + round(1/basicTF.fr);

end
