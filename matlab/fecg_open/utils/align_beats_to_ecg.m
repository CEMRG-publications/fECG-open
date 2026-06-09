% PURPOSE:  Fine-tune beat sample indices by snapping each to the nearest local
%           maximum (upright) or minimum (inverted) within a small search window.
%           This corrects ±few-sample timing errors introduced by the beat tracker.
% INPUTS:   beats         — 1×K integer, initial beat sample indices
%           ecg           — 1×N double, ECG waveform to search in
%           sample_window — scalar, half-width of search window (samples);
%                           5 for mbeats, 2 for fbeats (in main)
% OUTPUTS:  beats_new    — 1×K integer, refined beat sample indices
%           changed_ctr  — scalar, count of beats that were actually moved
% METHOD:   Determine polarity from mean(ecg(beats)); find local max or min in
%           [beat−window, beat+window]; update index if improvement found.
function [beats_new, changed_ctr] = align_beats_to_ecg(beats, ecg, sample_window)

    beats_new = beats;
    Po = mean(ecg(beats)) > 0; % polarity
    changed_ctr = 0;

    for i = 1:length(beats_new)
        beat_idx = beats_new(i);

        window_lower = max(beat_idx - sample_window, 1);
        window_upper = min(beat_idx + sample_window, length(ecg));

        if Po > 0 % signal upright: find max
            [max_val, max_idx] = max(ecg(window_lower : window_upper));
            max_idx = max_idx + beat_idx - sample_window - 1;
            if max_val > ecg(beat_idx)
                beats_new(i) = max_idx;
                changed_ctr = changed_ctr + 1;
            end
        else % signal inverted: find min
            [min_val, min_idx] = min(ecg(window_lower : window_upper));
            min_idx = min_idx + beat_idx - sample_window - 1;
            if min_val < ecg(beat_idx)
                beats_new(i) = min_idx;
                changed_ctr = changed_ctr + 1;
            end
        end
    end

end
