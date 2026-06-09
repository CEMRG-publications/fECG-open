% PURPOSE:  Detect beat times in an onset-strength signal using dynamic-programming
%           beat tracking guided by a time-varying tempo estimate.
% INPUTS:   onset — 1×N double, onset strength envelope (e.g. resampled ECG at osr Hz)
%           osr   — scalar, frame rate of onset (samples/s), here 100 Hz
%           tempo — 1×N double, expected instantaneous tempo (beats/sample) at each frame;
%                   derived from CurveExt_M ridge converted via basicTF.fr
%           alpha — scalar >=0, log-Gaussian transition cost weight (default 100)
% OUTPUTS:  beats — 1×K integer, detected beat frame indices (at osr)
% METHOD:   DP: cumscore(t) = max_{t'} [cumscore(t') − α·|log(prange/period)|²] + onset(t);
%           backtrace from global maximum. Prepends 1000 silence frames as initialisation.
function beats = beat_simple(onset, osr, tempo, alpha)
% beats = beat_simple(onset, osr, tempo, alpha)
% Core of the DP-based beat tracker.
% onset  : onset strength envelope at frame rate osr
% tempo  : target tempo in beats/sample (cycles per STFT frame at osr Hz),
%           NOT in BPM. Values are on the order of 0.01–0.04, constructed
%           by callers as HR_index * basicTF.fr where basicTF.fr = 0.02
%           cycles/sample at the internal 100 Hz rate. See outer header
%           (line 5) which correctly describes the unit as "beats/sample".
% alpha  : weight applied to transition cost
% beats  : chosen beat sample times (in samples)
% 2007-06-19 Dan Ellis dpwe@ee.columbia.edu
% 2016-08-13 Revised by Li Su

    if nargin < 4; alpha = 100; end

    onset = [zeros(1, 1000) onset];
    tempo = [ones(1, 1000) tempo];

    localscore = onset;
    backlink = -ones(1, length(localscore));
    cumscore = zeros(1, length(localscore));

    for i = 1001:length(localscore)
        period = (1/tempo(i))*osr;
        prange = round(-2*period):-round(period/2);
        txwt = (-alpha*abs((log(prange/-period)).^2));
        timerange = i + prange;
        scorecands = txwt + cumscore(timerange);
        [vv, xx] = max(scorecands);
        cumscore(i) = vv + localscore(i);
        backlink(i) = timerange(xx);
    end

    [vv, beats] = max(cumscore);

    while backlink(beats(1)) > 0
        beats = [backlink(beats(1)), beats];
    end
    beats = beats - 1000;
    beats = beats(beats > 0);

end
