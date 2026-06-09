% PURPOSE:  Offline Pan-Tompkins QRS detector with search-back for missed beats.
%           Detects R-peaks as the dominant peaks in a band-passed, differentiated,
%           squared, and integrated version of the ECG.
% INPUTS:   ecg        — 1×N or N×1 double, raw ECG (mV scale recommended)
%           varargin   — optional positional args: REF_PERIOD (0.25 s), THRES (0.3),
%                        fs (1000 Hz), fid_vec, SIGN_FORCE, debug (0), WIN_SAMP_SZ (7)
% OUTPUTS:  qrs_pos  — 1×K integer, R-peak sample indices
%           sign     — scalar, dominant polarity of the QRS complex (+1 or -1)
%           en_thres — scalar, energy threshold used for detection
% METHOD:   Sombrero-hat BPF → differentiate → square → integrate (MWI) → adaptive
%           threshold at 98th percentile → search-back in gaps > 1.5×median RRI.
function [qrs_pos, sign, en_thres] = qrs_detect2(ecg, varargin)
% QRS detector based on the P&T method. Offline implementation.
%
% inputs
%   ecg:        one ecg channel [mV]
%   varargin:   REF_PERIOD, THRES, fs, fid_vec, SIGN_FORCE, debug, WIN_SAMP_SZ
% outputs
%   qrs_pos:    indexes of detected peaks (samples)
%   sign:       sign of the peaks
%   en_thres:   energy threshold used

    WIN_SAMP_SZ = 7;
    REF_PERIOD = 0.250;
    THRES = 0.6;
    fs = 1000;
    fid_vec = [];
    SIGN_FORCE = [];
    debug = 0;

    switch nargin
        case 1
            % do nothing
        case 2
            REF_PERIOD = varargin{1};
        case 3
            REF_PERIOD = varargin{1};
            THRES = varargin{2};
        case 4
            REF_PERIOD = varargin{1};
            THRES = varargin{2};
            fs = varargin{3};
        case 5
            REF_PERIOD = varargin{1};
            THRES = varargin{2};
            fs = varargin{3};
            fid_vec = varargin{4};
        case 6
            REF_PERIOD = varargin{1};
            THRES = varargin{2};
            fs = varargin{3};
            fid_vec = varargin{4};
            SIGN_FORCE = varargin{5};
        case 7
            REF_PERIOD = varargin{1};
            THRES = varargin{2};
            fs = varargin{3};
            fid_vec = varargin{4};
            SIGN_FORCE = varargin{5};
            debug = varargin{6};
        case 8
            REF_PERIOD = varargin{1};
            THRES = varargin{2};
            fs = varargin{3};
            fid_vec = varargin{4};
            SIGN_FORCE = varargin{5};
            debug = varargin{6};
            WIN_SAMP_SZ = varargin{7};
        otherwise
            error('qrs_detect: wrong number of input arguments \n');
    end

    [a, b] = size(ecg);
    if (a > b); NB_SAMP = a; elseif (b > a); NB_SAMP = b; ecg = ecg'; end
    tm = 1/fs:1/fs:ceil(NB_SAMP/fs);

    MED_SMOOTH_NB_COEFF = round(fs/100);
    INT_NB_COEFF = round(WIN_SAMP_SZ*fs/256);
    SEARCH_BACK = 1;
    MAX_FORCE = [];
    MIN_AMP = 0.1;
    NB_SAMP = length(ecg);

    try
        b1 = [-7.757327341237223e-05  -2.357742589814283e-04 -6.689305101192819e-04 -0.001770119249103 ...
             -0.004364327211358 -0.010013251577232 -0.021344241245400 -0.042182820580118 -0.077080889653194 ...
             -0.129740392318591 -0.200064921294891 -0.280328573340852 -0.352139052257134 -0.386867664739069 ...
             -0.351974030208595 -0.223363323458050 0 0.286427448595213 0.574058766243311 ...
             0.788100265785590 0.867325070584078 0.788100265785590 0.574058766243311 0.286427448595213 0 ...
             -0.223363323458050 -0.351974030208595 -0.386867664739069 -0.352139052257134 ...
             -0.280328573340852 -0.200064921294891 -0.129740392318591 -0.077080889653194 -0.042182820580118 ...
             -0.021344241245400 -0.010013251577232 -0.004364327211358 -0.001770119249103 -6.689305101192819e-04 ...
             -2.357742589814283e-04 -7.757327341237223e-05];

        b1 = resample(b1, fs, 250);
        bpfecg = filtfilt(b1, 1, ecg)';

        if (sum(abs(ecg-median(ecg)) > MIN_AMP)/NB_SAMP) > 0.05

            dffecg = diff(bpfecg');
            sqrecg = dffecg.*dffecg;
            intecg = filter(ones(1, INT_NB_COEFF), 1, sqrecg);
            mdfint = medfilt1(intecg, MED_SMOOTH_NB_COEFF);
            delay = ceil(INT_NB_COEFF/2);
            mdfint = circshift(mdfint, -delay);

            if isempty(fid_vec); mdfintFidel = mdfint; else mdfintFidel(fid_vec > 2) = 0; end

            if NB_SAMP/fs > 90; xs = sort(mdfintFidel(fs:fs*90)); else xs = sort(mdfintFidel(fs:end)); end

            if isempty(MAX_FORCE)
                if NB_SAMP/fs > 10
                    ind_xs = ceil(98/100*length(xs));
                    en_thres = xs(ind_xs);
                else
                    ind_xs = ceil(99/100*length(xs));
                    en_thres = xs(ind_xs);
                end
            else
                en_thres = MAX_FORCE;
            end

            poss_reg = mdfint > (THRES*en_thres);

            if isempty(poss_reg); poss_reg(10) = 1; end

            if SEARCH_BACK
                indAboveThreshold = find(poss_reg);
                RRv = diff(tm(indAboveThreshold));
                medRRv = median(RRv(RRv > 0.01));
                indMissedBeat = find(RRv > 1.5*medRRv);
                indStart = indAboveThreshold(indMissedBeat);
                indEnd = indAboveThreshold(indMissedBeat+1);

                for i = 1:length(indStart)
                    poss_reg(indStart(i):indEnd(i)) = mdfint(indStart(i):indEnd(i)) > (0.5*THRES*en_thres);
                end
            end

            left  = find(diff([0 poss_reg']) == 1);
            right = find(diff([poss_reg' 0]) == -1);

            if SIGN_FORCE
                sign = SIGN_FORCE;
            else
                nb_s = length(left < 30*fs);
                loc = zeros(1, nb_s);
                for j = 1:nb_s
                    [~, loc(j)] = max(abs(bpfecg(left(j):right(j))));
                    loc(j) = loc(j) - 1 + left(j);
                end
                sign = mean(ecg(loc));
            end

            compt = 1;
            NB_PEAKS = length(left);
            maxval = zeros(1, NB_PEAKS);
            maxloc = zeros(1, NB_PEAKS);
            for i = 1:NB_PEAKS
                if sign > 0
                    [maxval(compt), maxloc(compt)] = max(ecg(left(i):right(i)));
                else
                    [maxval(compt), maxloc(compt)] = min(ecg(left(i):right(i)));
                end
                maxloc(compt) = maxloc(compt) - 1 + left(i);

                if compt > 1
                    if maxloc(compt)-maxloc(compt-1) < fs*REF_PERIOD && abs(maxval(compt)) < abs(maxval(compt-1))
                        maxloc(compt) = []; maxval(compt) = [];
                    elseif maxloc(compt)-maxloc(compt-1) < fs*REF_PERIOD && abs(maxval(compt)) >= abs(maxval(compt-1))
                        maxloc(compt-1) = []; maxval(compt-1) = [];
                    else
                        compt = compt + 1;
                    end
                else
                    compt = compt + 1;
                end
            end

            qrs_pos = maxloc;
            R_t = tm(maxloc);
            R_amp = maxval;
            hrv = 60./diff(R_t);
        else
            qrs_pos = [];
            R_t = [];
            R_amp = [];
            hrv = [];
            sign = [];
            en_thres = [];
        end
    catch ME
        rethrow(ME);
        for enb = 1:length(ME.stack); disp(ME.stack(enb)); end
        qrs_pos = [1 10 20]; sign = 1; en_thres = 0.5;
    end

end

%%
% > ⚠️ **Uncertainty (new):** Variables `R_t = tm(maxloc)`, `R_amp = maxval`, and `hrv = 60./diff(R_t)` are computed inside the detection loop but served no purpose except in the now-deleted plotting block.
% > **Inference:** Dead assignments with no functional impact; can be removed in a future pass.
