% PURPOSE:  Extract the maternal heart-rate curve from a synchrosqueezed TFR by
%           masking the TFR to the physiological maternal HR band (0.4–2.4 Hz at
%           basicTF.fs=100 Hz, i.e. 24–144 bpm) and applying dynamic-programming
%           ridge extraction.
% INPUTS:   rtfr_post — freq×time double, synchrosqueezed TFR
%           basicTF   — struct: basicTF.fr is the frequency resolution
%           dtw       — scalar, DP smoothness weight (= lam_curve, 50 or 5 in main)
% OUTPUTS:  HR — 1×time double, estimated maternal HR as frequency-bin index per frame
% METHOD:   Zero TFR outside [0.4/fr, 2.4/fr] bin range → CurveExt_M (DP ridge).
function HR = Extract_mhr(rtfr_post, basicTF, dtw)

    rtfr_post(1:round(0.4/basicTF.fr), :) = 0;
    rtfr_post(round(2.4/basicTF.fr):end, :) = 0;
    HR = CurveExt_M(rtfr_post(1:round(2.4/basicTF.fr), :)', dtw);

end
