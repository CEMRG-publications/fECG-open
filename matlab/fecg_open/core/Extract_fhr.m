% NOTE: Python equivalent is named extract_fhr (lowercase) in fecg_open/core/extract_fhr.py (PEP 8 rename).
% PURPOSE:  Extract the fetal heart-rate curve from a synchrosqueezed TFR via two
%           stages with different frequency bounds:
%             1) Masking (0.4-3.2 Hz): rows of the full TFR outside this band are
%                zeroed to suppress DC drift/baseline wander (<0.4 Hz) and
%                high-frequency noise (>3.2 Hz). The retained band still contains
%                maternal HR energy (maternal band: 0.4-2.4 Hz).
%             2) DP ridge search (1.0-3.2 Hz): a 1.0-3.2 Hz submatrix of the masked
%                TFR is passed to CurveExt_M for dynamic-programming ridge tracking.
%                The lower bound is raised to 1.0 Hz (from the 0.4 Hz masking bound)
%                to exclude the maternal fundamental and its lower harmonics, which
%                would otherwise contaminate the fetal ridge estimate. Returned bin
%                indices are corrected back to full-TFR coordinates by adding
%                round(1/basicTF.fr).
% INPUTS:   rtfr_post — freq×time double, synchrosqueezed TFR (after maternal suppression)
%           basicTF   — struct: basicTF.fr is the frequency resolution
%           dtw       — scalar, DP smoothness weight (= lam_curve)
% OUTPUTS:  HR — 1×time double, estimated fetal HR as frequency-bin index per frame
function HR = Extract_fhr(rtfr_post, basicTF, dtw)

    rtfr_post(1:round(0.4/basicTF.fr), :) = 0;
    rtfr_post(round(3.2/basicTF.fr):end, :) = 0;

    HR = CurveExt_M(rtfr_post(round(1/basicTF.fr)+1:round(3.2/basicTF.fr), :)', dtw);
    HR = HR + round(1/basicTF.fr);

end
