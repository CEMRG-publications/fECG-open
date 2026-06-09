% PURPOSE:  Compute a time-frequency representation (TFR) of a 1-D signal using
%           the de-shape Synchrosqueezing Transform (SST). The SST sharpens the
%           STFT by reassigning energy to the instantaneous frequency, producing
%           a crisp ridge at the dominant heart-rate frequency at each time step.
% INPUTS:   x       — 1×N double, ECG-like signal (resampled to basicTF.fs Hz)
%           basicTF — struct: STFT window (win), hop, fs (100 Hz), freq resolution (fr)
%           advTF   — struct: threshold (ths), HighFreq, LowFreq (fraction of basicTF.fs)
%           cepR    — struct: cepstral exponent g (0.3), threshold Tc (0)
%           P       — struct: num_s=1 (harmonic order)
% OUTPUTS:  tfr    — freq×time double, STFT magnitude
%           ceps   — freq×time double, cepstral envelope (harmonic enhancer)
%           tceps  — freq×time double, cepstrum mapped to frequency axis
%           tfrr   — freq×time double, TFR after cepstral weighting
%           rtfr   — freq×time double, synchrosqueezed TFR (sharpened ridges)
%           tfrsq  — freq×time double, synchrosqueezed raw STFT (unweighted)
%           tfrtic — freq×1 double, frequency axis (cycles per sample at basicTF.fs)
%           t      — 1×time double, time axis (sample indices at basicTF.fs)
% METHOD:   1-tap Flattop-windowed STFT → cepstral harmonic weighting →
%           instantaneous frequency deviation (IFD) → synchrosqueezing (SST11)
function [tfr, ceps, tceps, tfrr, rtfr, tfrsq, tfrtic, t] = CFPH(x, basicTF, advTF, cepR, P)

    win = basicTF.win;
    hop = basicTF.hop;
    fs = basicTF.fs;
    fr = basicTF.fr;

    HighFreq = advTF.HighFreq;
    LowFreq = advTF.LowFreq;
    ths = advTF.ths;

    num_s = P.num_s;

    h = tftb_window(win);
    Dh = dwindow(h);
    Dho = dwindow(Dh);
    h = [h 20*Dh];
    Dh = [Dh 20*Dho];

    g = cepR.g;
    Tc = cepR.Tc;

    rv = [1 0];
    rh = rv * h';
    rDh = rv * Dh';
    h2 = rh'; Dh2 = rDh';

    [tfr, ifd, tfrtic, t] = STFT_IFD_fast(x, fr/fs, hop, h2, Dh2);
    tfr = abs(tfr);

    [ceps, tceps] = cepstrum_convert(tfr, tfrtic, g, fs, Tc, HighFreq, LowFreq);

    tfr0 = tfr;
    tfrr = tfr0.*tceps;
    tfrr(tfrr < 0) = 0;
    tfrr(1:round(LowFreq*fs/fr), :) = 0;

    tfrr = tfrr(1:round(HighFreq*fs*num_s/fr), :);
    ifd  = ifd(1:round(HighFreq*fs*num_s/fr), :);

    [~, tfr3]   = synchrosqueeze1win(tfrr, ifd, fr/fs, fr, HighFreq, fs, ths);
    [~, tfrsq]  = synchrosqueeze1win(tfr0, ifd, fr/fs, fr, HighFreq, fs, ths);
    rtfr = tfr3;

    tfrr   = tfrr(1:round(HighFreq*fs/fr), :);
    tceps  = tceps(1:round(HighFreq*fs/fr), :);
    tfr    = tfr(1:round(HighFreq*fs/fr), :);
    tfrtic = tfrtic(1:size(rtfr, 1));

end
