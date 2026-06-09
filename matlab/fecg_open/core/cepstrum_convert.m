% PURPOSE:  Convert an STFT magnitude to a cepstrum-based harmonic weighting mask
%           (tceps). The mask emphasises TFR bins whose energy pattern is consistent
%           with a periodic (harmonic) source — i.e. the ECG fundamental + harmonics.
% INPUTS:   tfr     — freq×time double, STFT magnitude
%           tfrtic  — freq×1 double, normalised frequency axis
%           g       — scalar, cepstral exponent (0.3); controls harmonic sharpness
%           fs      — scalar, internal sample rate (basicTF.fs = 100 Hz)
%           Tc      — scalar, cepstral threshold (0); values below are zeroed
%           HighFreq— scalar, upper cutoff as fraction of fs (= 10/100)
%           LowFreq — scalar, lower cutoff as fraction of fs (= 0.5/100)
% OUTPUTS:  ceps0 — ceps×time double, raw cepstrum coefficients
%           tceps — freq×time double, harmonic weighting mask mapped to frequency axis
% METHOD:   |tfr|^g → IFFT → zero outside [1/HighFreq, 1/LowFreq] quefrency bins →
%           interpolate → map to frequency axis via freq_scale = 10·fs / quefrency_index
function [ceps0, tceps] = cepstrum_convert(tfr, tfrtic, g, fs, Tc, HighFreq, LowFreq)

    ceps = real(ifft(abs(tfr).^g, 2*size(tfr, 1), 1));

    for mi = 1:size(ceps, 2)
        ceps(1:round(1/HighFreq), mi) = 0;
    end

    ceps(isnan(ceps) | isinf(ceps)) = 0;

    ceps = ceps(1:round(1/LowFreq), :);
    ceps0 = ceps;
    ceps = interp1(1:size(ceps, 1), ceps, 1:0.1:size(ceps, 1));
    tceps = zeros(length(tfrtic), size(ceps, 2));
    freq_scale = 10.*fs ./ (1:size(ceps, 1)-1);
    for ii = 2:length(tfrtic)-1
        p_index = find(freq_scale > (tfrtic(ii-1)+tfrtic(ii))*fs/2 & freq_scale < (tfrtic(ii+1)+tfrtic(ii))*fs/2);
        if ~isempty(p_index)
            % Each frequency bin is weighted equally (ii.^0 = 1), as adopted for the SampTA2017 manuscript.
            tceps(ii, :) = sum(ceps(p_index, :), 1) .* (ii.^0);
        end
    end
    tceps(tceps < Tc) = 0;

end
