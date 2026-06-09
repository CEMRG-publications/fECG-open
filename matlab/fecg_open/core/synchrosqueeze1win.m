% PURPOSE:  Apply the Synchrosqueezing Transform (SST) to reassign STFT energy
%           to the instantaneous frequency, sharpening spectral ridges. A triangular
%           smoothing window (Smooth=1, weight 0.25) is applied during accumulation.
% INPUTS:   tfr      — freq×time double, STFT magnitude (cepstrum-weighted)
%           ifd      — freq×time double, instantaneous frequency deviation matrix
%           alpha    — scalar, STFT frequency resolution (fr/basicTF.fs)
%           fr       — scalar, basicTF.fr (frequency resolution, cycles/sample)
%           HighFreq — scalar, upper frequency limit (fraction of basicTF.fs)
%           fs       — scalar, basicTF.fs (internal sampling rate, 100 Hz)
%           ths      — scalar, TFR threshold (entries below ths·mean are zeroed)
% OUTPUTS:  tfr  — freq×time double, truncated STFT (to HighFreq)
%           rtfr — freq×time double, synchrosqueezed TFR with triangular smoothing
% METHOD:   Round IFD → accumarray reassignment → triangular accumulation (Smooth=1)
function [tfr, rtfr] = synchrosqueeze1win(tfr, ifd, alpha, fr, HighFreq, fs, ths)

    omega = ifd;

    tfr   = tfr(1:round(HighFreq*fs/fr), :);
    omega = round(omega(1:round(HighFreq*fs/fr), :));

    [M, ~] = size(tfr);
    OrigIndex = repmat((1:M)', [1 size(tfr, 2)]);
    % Smooth=1: window half-width is 1; clamp indices that would go out of bounds
    omega(OrigIndex - omega < 3 | OrigIndex - omega > M-2) = 0;

    Ex = mean(sum(abs(tfr)));
    Threshold = ths*Ex;
    tfr(abs(tfr) < Threshold) = 0;

    totLength = size(tfr, 1)*size(tfr, 2);
    new_idx = (1:totLength)' - omega(:);

    % Smooth=1 hardcoded: triang(3)/sum(triang(3)) = [0.25;0.5;0.25], where 0.25 is the
    % edge weight and 0.5 is the center weight. Only the edge weight (0.25) is applied
    % at index new_idx. The loop "for ii=1:Smooth-1" runs zero iterations for Smooth=1,
    % so the center coefficient (0.5) and second edge (0.25) are never accumulated.
    % Net effect: the synchrosqueezed output is scaled by 0.25× rather than 1.0×.
    rtfr = accumarray([1; new_idx(2:totLength-1); totLength], 0.25.*tfr(:));
    rtfr = [rtfr; zeros(totLength-length(rtfr), 1)];
    rtfr = reshape(rtfr, size(tfr, 1), size(tfr, 2));

end