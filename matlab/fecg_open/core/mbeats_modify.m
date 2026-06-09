% PURPOSE:  Snap approximate R-peak locations (e.g. from fusion) to the nearest
%           true local maximum or minimum within a ±48-sample search window.
% INPUTS:   x_up   — 1×N double, signal in which to search for peaks
%           mlocsf — 1×K integer, initial approximate beat locations (samples)
% OUTPUTS:  mbeats_p — 1×K integer, refined R-peak (positive peak) indices
%           mbeats_q — 1×K integer, refined S-point (negative peak) indices
%           R_amp    — scalar, median R-peak amplitude
%           S_amp    — scalar, median S-point amplitude
%           Po       — ±1, signal polarity (1 if R > S, -1 if inverted)
% METHOD:   Local max/min search in ±SearchLen_p/q window; swap R/S if polarity inverted.
function [mbeats_p, mbeats_q, R_amp, S_amp, Po] = mbeats_modify(x_up, mlocsf)
    % Relocate mbeats to peaks in x_up

    x0_len = length(x_up);
    mbeats_p = [];
    mbeats_q = [];
    SearchLen_p = 48;
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
    Po = 1;
    if abs(median(x_up(mbeats_p))) < abs(median(x_up(mbeats_q)))
        [mbeats_q, mbeats_p] = deal(mbeats_p, mbeats_q);
        Po = -1;
        x_up = -x_up;
    end

    R_amp = median(x_up(mbeats_p));
    S_amp = median(x_up(mbeats_q));

end
