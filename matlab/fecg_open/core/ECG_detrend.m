% PURPOSE:  Remove slow baseline drift from an ECG signal using a cascaded
%           median-filter approach, optionally preserving beat morphology.
% INPUTS:   x0        — 1×N double, raw ECG signal
%           num_med_s — scalar, short window length (samples); 51 (no-morph) or 101 (morph)
%           num_med_l — scalar, long window length (samples); 301 in main
%           if_morph  — 0/1; 1 = cascaded median (gentle, keeps beat shape);
%                              0 = single median (aggressive, removes morphology)
% OUTPUTS:  x0 — 1×N double, baseline-removed ECG signal
% METHOD:   Short movmedian → (optional) long movmedian → loess smooth (span=10);
%           subtract the resulting trend from the original signal.
function x0 = ECG_detrend(x0, num_med_s, num_med_l, if_morph)
    % detrend
    % if_morph = 1 : preserve morphology
    %           = 0 : otherwise
    if if_morph
        x0_trend = movmedian(x0, num_med_s);
        x0_trend = movmedian(x0_trend, num_med_l);
    else
        x0_trend = movmedian(x0, num_med_s);
    end

    x0_trend = smooth(x0_trend, 10, 'loess')';
    x0 = x0 - x0_trend;

end
