% PURPOSE:  Remove spurious beat detections that violate the physiological
%           refractory period (minimum inter-beat interval).
% INPUTS:   beats — 1×K integer, detected beat frame indices
%           xf    — 1×N double, signal (used to compare adjacent peak amplitudes)
%           fs    — scalar, sampling rate of xf
%           RS    — scalar, minimum allowed RR interval (s); 0.25 s in main
% OUTPUTS:  y — 1×M integer, filtered beat indices (M ≤ K)
% METHOD:   For any pair of beats closer than RS·fs samples, discard the one
%           with the smaller absolute amplitude; uses setdiff for removal.
function y = RRconstraint(beats, xf, fs, RS)

    RRI = diff(beats);
    FP = [];

    for ii = 1:length(RRI)
        if RRI(ii) <= RS*fs
            % take the "larger peak" as the R peak
            if abs(xf(beats(ii))) <= abs(xf(beats(ii+1)))
                FP = [FP beats(ii)];
            else
                FP = [FP beats(ii+1)];
            end
        end
    end
    y = setdiff(beats, FP);

end
