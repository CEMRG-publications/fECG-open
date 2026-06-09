% PURPOSE:  Compute the Short-Time Fourier Transform and the Instantaneous Frequency
%           Deviation (IFD) needed for Synchrosqueezing. The IFD measures how much
%           each frequency bin's energy is offset from its nominal frequency — it is
%           used by synchrosqueeze1win to reassign STFT energy to sharper ridges.
% INPUTS:   x     — 1×N double, signal at basicTF.fs (100 Hz)
%           alpha — scalar, frequency resolution (basicTF.fr / basicTF.fs = 0.02/100)
%           Hop   — scalar, STFT hop size in samples (basicTF.hop = 20)
%           h     — L×1 double, analysis window (Flattop, possibly padded with Dh)
%           Dh    — L×1 double, window derivative (for IFD computation)
% OUTPUTS:  tfr    — freq×time complex, STFT (lower half of spectrum)
%           ifd    — freq×time double, instantaneous frequency deviation per bin
%           tfrtic — freq×1 double, normalised frequency axis [0, 0.5]
%           t      — 1×time integer, centre-sample indices of each STFT frame
% METHOD:   Frame signal with h; compute FFT; IFD = Im{N·(Dh-STFT)/(h-STFT)/(2π)}
% Synchrosqueezing by Li Su, 2015
function [tfr, ifd, tfrtic, t] = STFT_IFD_fast(x, alpha, Hop, h, Dh)

    if size(h, 2) > size(h, 1)
        h = h';
    end
    if size(Dh, 2) > size(Dh, 1)
        Dh = Dh';
    end

    N = length(-0.5+alpha:alpha:0.5);
    Win_length = max(size(h));
    TH = 7*N/size(h, 1);
    tfrtic = linspace(0, 0.5, round(N/2))';

    Overlap = Win_length - Hop;
    Lh = floor((Win_length-1)/2);
    t = Hop:Hop:floor(length(x)/Hop)*Hop;
    x_Frame = zeros(N, length(t));
    tf2 = zeros(N, length(t));

    for ii = 1:length(t)
        ti = t(ii);
        tau = -min([round(N/2)-1, Lh, ti-1]) : min([round(N/2)-1, Lh, length(x)-ti]);
        indices = rem(N+tau, N) + 1;
        norm_h = norm(h(Lh+1+tau));
        x_Frame(indices, ii) = (x(ti+tau) - mean(x(ti+tau)))' .* conj(h(Lh+1+tau)) / norm_h;
        tf2(indices, ii) = (x(ti+tau) - mean(x(ti+tau)))' .* conj(Dh(Lh+1+tau)) / norm_h;
    end

    Stime = round(Hop - Win_length/2 + 1) + ceil(Overlap/Hop/2)*Hop;

    tfr = fft(x_Frame, N, 1);
    tfr = tfr(1:round(N/2), :);

    tf2 = fft(tf2, N, 1);
    tf2 = tf2(1:round(N/2), :);

    omega = zeros(size(tf2));
    avoid_warn = find(tfr ~= 0);
    omega(avoid_warn) = imag(N*tf2(avoid_warn) ./ tfr(avoid_warn) / (2.0*pi));
    ifd = omega;

end
