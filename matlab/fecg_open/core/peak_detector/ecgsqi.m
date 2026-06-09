% PURPOSE:  Compute a windowed F1-based Signal Quality Index (SQI) measuring agreement
%           between two QRS annotation sequences. Higher SQI → detectors agree → clean signal.
% INPUTS:   ann1     — K×1 double, beat times from detector 1 (seconds)
%           ann2     — M×1 double, beat times from detector 2 (seconds)
%           THR      — scalar, matching tolerance (s); 0.150 s in main
%           SIZE_WIND— scalar, SQI window size (s); 4 s in main
%           REG_WIN  — scalar, SQI evaluation stride (s); 1 s in main
%           LG_MED   — scalar, number of adjacent windows for min-smoothing; 0 in main
%           LG_REC   — scalar, total record length (s)
%           N_WIN    — scalar, total number of SQI windows
% OUTPUTS:  sqi  — N_WIN×1 double, SQI per window (0=no agreement, 1=perfect)
%           tsqi — N_WIN×1 double, window start times (s)
% METHOD:   For each window, compute F1 = min(precision, recall) between ann1 and ann2
%           using histc-based ±THR matching; optionally smooth with min-filter (LG_MED).
function [sqi, tsqi] = ecgsqi(ann1, ann2, THR, SIZE_WIND, REG_WIN, LG_MED, LG_REC, N_WIN)

    if nargin < 7
        error('ecgsqi:notEnoughArguments', 'ecgsqi.m received %d arguments, expected 7.', nargin)
    end

    xi = [ann1' - THR; ann1' + THR];
    xi = xi(:);

    idxFix = [false; diff(xi) < 0];
    xi_fixed = [xi(idxFix), xi([idxFix(2:end); false])];
    xi_fixed = mean(xi_fixed, 2);
    xi(idxFix) = xi_fixed;
    xi([idxFix(2:end); false]) = xi_fixed;

    N_J = histc(ann2, xi);

    xi = [ann2' - THR; ann2' + THR];
    xi = xi(:);
    idxFix = [false; diff(xi) < 0];
    xi_fixed = [xi(idxFix), xi([idxFix(2:end); false])];
    xi_fixed = mean(xi_fixed, 2);
    xi(idxFix) = xi_fixed;
    xi([idxFix(2:end); false]) = xi_fixed;
    N_G = histc(ann1, xi);

    N_J = N_J(1:2:end);
    N_G = N_G(1:2:end);
    N_J = N_J(:);
    N_G = N_G(:);

    xi1 = (0:REG_WIN:LG_REC)';
    xi2 = xi1 + SIZE_WIND;

    xi1_trunc = xi1;
    xi1_trunc(xi2 > LG_REC) = LG_REC - SIZE_WIND;

    F1_1 = zeros(N_WIN, 1);
    F1_2 = zeros(N_WIN, 1);

    for w = 1:numel(xi1_trunc)
        idx1 = ann1 > xi1_trunc(w) & ann1 < xi2(w);
        idx2 = ann2 > xi1_trunc(w) & ann2 < xi2(w);
        F1_1(w) = mean(N_J(idx1) == 1);
        F1_2(w) = mean(N_G(idx2) == 1);
    end

    F1 = min(F1_1, F1_2);
    F1(isnan(F1)) = 0;

    idxRem = xi1 >= LG_REC;
    F1(idxRem) = [];
    xi1(idxRem) = [];

    if size(F1, 1) < (LG_MED*2+1)
        F1smooth = F1;
    else
        F1smooth = nan(size(F1, 1), 2*LG_MED+1);
        for k = 1:LG_MED
            F1smooth(:, k) = vertcat(ones(k, 1), F1(1:end-k));
            F1smooth(:, k+LG_MED) = vertcat(F1(k+1:end), ones(k, 1));
        end
        F1smooth(:, end) = F1;
        F1smooth = min(F1smooth, [], 2);
    end
    sqi = F1smooth;
    tsqi = xi1;

end
