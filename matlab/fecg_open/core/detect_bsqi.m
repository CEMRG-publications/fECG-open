% PURPOSE:  Compute a beat-by-beat Signal Quality Index (bSQI) for an ECG signal
%           by comparing two independent QRS detectors: jqrs (energy-based, runs
%           internally) and a provided reference annotation (acting as gqrs proxy).
%           Returns the per-window F1 score between the two detector outputs.
% INPUTS:   data   — N×1 double, ECG signal (column vector)
%           header — 1×1 cell {'ECG'}, signal type label
%           fs     — scalar, sampling rate (Hz)
%           opt    — struct: SQI window/threshold parameters (see setDetectOptions)
%           beats  — 1×M integer, reference R-peak indices (gqrs proxy, in samples)
% OUTPUTS:  qrs        — 1×K double, final beat times (s) after optional switching
%           sqi        — 1×1 cell, per-window SQI vector (F1 scores, 0–1)
% METHOD:   run_qrsdet_by_seg_ali (jqrs) → ecgsqi (F1 between jqrs and gqrs per window)
function [qrs, sqi] = detect_bsqi(data, header, fs, opt, beats)

    % option setting
    if nargin < 4
        [opt] = setDetectOptions;
    else
        if ~isstruct(opt)
            error('detect:invalidOptions', 'Second argument should be a structure.');
        else
            [opt] = setDetectOptions(opt);
        end
    end

    opt.LG_REC = size(data, 1) ./ fs;
    opt.N_WIN = ceil(opt.LG_REC/opt.REG_WIN);

    % getSignalIndices
    idxECG = cellfun(@any, regexpi(header, 'ecg', 'once'));
    idxECG = find(idxECG);
    idxECG = idxECG(:)';

    if numel(fs) == 1
        fs = repmat(fs, 1, numel(header));
    end

    % INITIALISE
    [~, M] = size(data);
    ann_jqrs = cell(1, M);
    ann_gqrs = cell(1, M);
    sqi_ecg  = cell(1, M);

    % ECG PEAK DETECT
    if ~isempty(idxECG)
        for m = idxECG
            ann_jqrs{m} = run_qrsdet_by_seg_ali(data(:, m), fs(m), opt);

            ann_gqrs_m = beats(m, :);
            ann_gqrs{m} = ann_gqrs_m(ann_gqrs_m ~= 0);

            ann_jqrs{m} = ann_jqrs{m}(:) ./ fs(m);
            ann_gqrs{m} = ann_gqrs{m}(:) ./ fs(m);
        end
    end

    % ECG LEAD WISE SQI
    for m = idxECG
        if ~isempty(ann_gqrs{m}) && ~isempty(ann_jqrs{m})
            [sqi_ecg{m}, ~] = ecgsqi(ann_gqrs{m}, ann_jqrs{m}, ...
                opt.THR, opt.SIZE_WIND, opt.REG_WIN, opt.LG_MED, ...
                opt.LG_REC, opt.N_WIN);
        else
            sqi_ecg{m} = zeros(opt.N_WIN(m), 1);
        end
    end

    % set up for switching (ECG only)
    qrs_comp   = ann_gqrs(idxECG);
    sqi        = sqi_ecg(idxECG);
    qrs = qrs_comp{1};

end
