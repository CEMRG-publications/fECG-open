% SOURCE: https://raw.githubusercontent.com/alistairewj/peak-detector/master/sources/setDetectOptions.m
function [opt] = setDetectOptions(varargin)
% setDetectOptions  Build and validate the options struct for beat detection and SQI scoring.
%
% PURPOSE
%   Returns a struct of parameters used by the jQRS/gQRS beat detector and
%   the beat-quality index (bSQI) scorer.  When called with no arguments,
%   returns factory defaults.  When called with a struct, merges the
%   supplied fields into the defaults after field-by-field validation.
%
% INPUTS
%   varargin{1} - (optional struct) partial options struct.  Recognised
%                 fields and their defaults:
%                   SIZE_WIND             10   (even integer) SQI window, seconds
%                   LG_MED                 3   (integer)      SQI median-filter half-width
%                   REG_WIN                1   (integer)      jQRS regularisation window
%                   THR                0.150   (numeric)      jQRS beat-detection threshold
%                   SQI_THR              0.8   (0-1)          minimum bSQI to accept a segment
%                   JQRS_THRESH          0.3   (numeric)      jQRS peak-detection threshold
%                   JQRS_REFRAC         0.25   (numeric)      jQRS refractory period (s)
%                   JQRS_INTWIN_SZ         7   (integer)      jQRS integration window size
%                   JQRS_WINDOW           15   (integer)      jQRS search window (s)
%                   ENABLE_OTHER_DETECTORS 0   (0|1)          enable auxiliary detectors
%                   SAVE_STUFF             0   (0|1)          save intermediate detector output
%                 Unrecognised fields are silently ignored.
%
% OUTPUTS
%   opt - (struct) validated options struct.  Includes all fields above plus
%         the derived field HALF_WIND = SIZE_WIND/2.
%
%   Cite: Johnson et al. (2015), Physiological Measurement 36:1665-1677.

    opt_default.SIZE_WIND = 10;
    opt_default.LG_MED = 3;
    opt_default.REG_WIN = 1;
    opt_default.THR = 0.150;
    opt_default.SQI_THR = 0.8;
    opt_default.JQRS_THRESH = 0.3;
    opt_default.JQRS_REFRAC = 0.25;
    opt_default.JQRS_INTWIN_SZ = 7;
    opt_default.JQRS_WINDOW = 15;
    opt_default.ENABLE_OTHER_DETECTORS = 0;
    opt_default.SAVE_STUFF = 0;

    if nargin == 0
        opt = opt_default;
        return;
    elseif nargin == 1
        opt = varargin{1};
    else
        error('Incorrect number of inputs.');
    end

    if nargin > 0 && isstruct(opt)
        fn = fieldnames(opt);
        fn_default = fieldnames(opt_default);
        for f = 1:numel(fn)
            if ismember(fn{f}, fn_default)
                val = opt.(fn{f});

                switch fn{f}
                    case {'SAVE_STUFF', 'ENABLE_OTHER_DETECTORS'}
                        if ~isnumeric(val) || (val ~= 0 && val ~= 1)
                            error('setDetectOptions:badValue', '%s: field can only take values 0 or 1.', fn{f});
                        end
                    case {'THR', 'JQRS_THRESH', 'JQRS_REFRAC'}
                        if ~isnumeric(val)
                            error('setDetectOptions:badValue', '%s: field should be an integer.', fn{f});
                        end
                    case {'SQI_THR'}
                        if ~isnumeric(val) || val < 0 || val > 1
                            error('setDetectOptions:badValue', '%s: field should be an integer.', fn{f});
                        end
                    case {'LG_MED', 'REG_WIN', 'JQRS_INTWIN_SZ', 'JQRS_WINDOW'}
                        if ~isnumeric(val) || round(val) ~= val
                            error('setDetectOptions:badValue', '%s: field should be an integer.', fn{f});
                        end
                    case {'SIZE_WIND'}
                        if ~isnumeric(val) || round(val) ~= val || mod(val, 2) ~= 0
                            error('setDetectOptions:badValue', '%s: field should be an integer divisible by two.', fn{f});
                        end
                end

                opt_default.(fn{f}) = val;
            else
                fprintf('Ignoring unrecognized option %s\n', fn{f});
            end
        end
    end

    opt_default.HALF_WIND = opt_default.SIZE_WIND/2;
    opt = opt_default;

end
