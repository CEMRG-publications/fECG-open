function [peakI, peakAmp] = findQRSpeaks(ecg, win)
% findQRSpeaks  Detect local maxima in an ECG signal using a sliding window.
%
% PURPOSE
%   Scans the input signal and identifies samples that are strictly the
%   maximum within a symmetric window of half-width win.  Used as a
%   lightweight QRS candidate detector before SQI-based beat refinement.
%
% INPUTS
%   ecg     - (1 x L or L x 1 real) single-channel ECG signal.
%   win     - (scalar positive integer) half-width of the search window in
%             samples.  A sample at index i is a local maximum if it is the
%             maximum of ecg(i-win : i+win).
%
% OUTPUTS
%   peakI   - (1 x P integer) sample indices of detected local maxima.
%   peakAmp - (1 x P real) ECG amplitude at each detected peak.

    L = length(ecg);
    peakI = zeros(size(ecg));
    for i = (1+win):(L-win)
        [M, I] = max(ecg(i-win:i+win));
        if i == (i-win-1+I)
            peakI(i) = 1;
        end
    end
    peakI = find(peakI > 0);
    peakAmp = ecg(peakI);

end
