% PURPOSE:  Run the jqrs P&T-based QRS detector on non-overlapping windows of an ECG,
%           forcing consistent peak polarity across windows to avoid alternating detections.
% INPUTS:   ecg — N×1 double, ECG signal
%           fs  — scalar, sampling rate (Hz)
%           opt — struct: JQRS_WINDOW (15 s), JQRS_THRESH (0.3), JQRS_REFRAC (0.25 s), JQRS_INTWIN_SZ (7)
% OUTPUTS:  QRS — 1×K integer, detected R-peak sample indices
% METHOD:   Divide into JQRS_WINDOW-second segments; run qrs_detect2 on each with
%           1-second overlap; concatenate results; propagate sign to next segment.
function QRS = run_qrsdet_by_seg_ali(ecg, fs, opt)
% Run QRS detector segment-by-segment (non-overlapping) to handle large
% artefacts at record boundaries.
%   Original file: https://raw.githubusercontent.com/alistairewj/peak-detector/master/sources/run_qrsdet_by_seg_ali.m
%
% inputs
%   ecg:    ECG signal
%   fs:     sampling frequency
%   opt:    options struct (jqrs parameters)
% output
%   QRS:    QRS locations in samples

    if nargin < 1; error('run_qrsdet_by_seg: wrong number of input arguments \n'); end
    if nargin < 2; fs = 1000; end
    if nargin < 3 || ~isstruct(opt)
        opt = setOptions;
    else
        [opt] = setOptions(opt);
    end

    segsizeSamp = opt.JQRS_WINDOW*fs;
    NbSeg = floor(length(ecg)/segsizeSamp);
    QRS = cell(NbSeg, 1);
    signForce = 0;

    if NbSeg == 0
        QRS = 1;
        return;
    end

    try
        % First subsegment - look forward 1s
        dTplus = fs;
        dTminus = 0;
        start = 1;
        stop = segsizeSamp;

        if NbSeg == 1
            dTplus = 0;
            stop = length(ecg);
        end

        [QRStemp, signForce] = qrs_detect2(ecg(start-dTminus:stop+dTplus), opt.JQRS_REFRAC, opt.JQRS_THRESH, fs, [], signForce, 0, opt.JQRS_INTWIN_SZ);
        QRS{1} = QRStemp(:);

        start = start + segsizeSamp;
        stop = stop + segsizeSamp;

        for ch = 2:NbSeg-1
            dTplus = fs;
            dTminus = fs;

            [QRStemp, signForce] = qrs_detect2(ecg(start-dTminus:stop+dTplus), opt.JQRS_REFRAC, opt.JQRS_THRESH, fs, [], signForce);

            NewQRS = (start-1) - dTminus + QRStemp;
            NewQRS(NewQRS > stop) = [];
            NewQRS(NewQRS < start) = [];

            if ~isempty(NewQRS) && ~isempty(QRS{ch-1})
                NewQRS(NewQRS < QRS{ch-1}(end)) = [];
                if ~isempty(NewQRS) && (NewQRS(1) - QRS{ch-1}(end)) < opt.JQRS_REFRAC*fs
                    NewQRS(1) = [];
                end
            end
            QRS{ch} = NewQRS(:);

            start = start + segsizeSamp;
            stop = stop + segsizeSamp;
        end

        if NbSeg > 1
            ch = NbSeg;
            stop = length(ecg);
            dTplus = 0;
            dTminus = fs;
            [QRStemp, signForce] = qrs_detect2(ecg(start-dTminus:stop+dTplus), opt.JQRS_REFRAC, opt.JQRS_THRESH, fs, [], signForce);

            NewQRS = (start-1) - dTminus + QRStemp;
            NewQRS(NewQRS > stop) = [];
            NewQRS(NewQRS < start) = [];

            if ~isempty(NewQRS) && ~isempty(QRS{ch-1})
                NewQRS(NewQRS < QRS{ch-1}(end)) = [];
                if ~isempty(NewQRS) && (NewQRS(1) - QRS{ch-1}(end)) < opt.JQRS_REFRAC*fs
                    NewQRS(1) = [];
                end
            end
            QRS{ch} = NewQRS(:);
        end

        QRS = vertcat(QRS{:});

    catch ME
        for enb = 1:length(ME.stack)
            disp(ME.stack(enb))
        end
        rethrow(ME);
    end

end


function [opt] = setOptions(opt)
    opt_default.JQRS_THRESH = 0.3;
    opt_default.JQRS_WINDOW = 15;
    opt_default.JQRS_REFRAC = 0.250;
    opt_default.JQRS_INTWIN_SZ = 7;
    if nargin > 0 && isstruct(opt)
        fn = fieldnames(opt);
        fn_default = fieldnames(opt_default);
        for f = 1:numel(fn)
            if ismember(fn{f}, fn_default)
                opt_default.(fn{f}) = opt.(fn{f});
            end
        end
    end
    opt = opt_default;
end
