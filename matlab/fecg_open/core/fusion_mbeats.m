% PURPOSE:  Fuse R-peak lists from multiple signal orientations into a single
%           consensus beat sequence by scoring each orientation's reliability and
%           clustering nearby detections from the surviving orientations.
% INPUTS:   mbeats_c — 1×M cell, each cell is a 1×K_i integer vector of R-peak indices
%           aECG_c   — 1×M cell, each cell is a 1×N double signal for the orientation
%           fs       — scalar, sampling rate (Hz)
% OUTPUTS:  mbeats — 1×K integer, fused and rounded R-peak sample indices
% METHOD:   Stage 1 — filter orientations with implausible RRI (0.4–1.6 s);
%           score remaining by template correlation and RRI regularity.
%           Stage 2 — keep orientations scoring ≥ 50% of best score, refine
%           beat sets by correlation threshold (th3_3=0.5).
%           Stage 3 — cluster peaks within 30 ms; keep clusters present in
%           ≥ nb_c/2 orientations; report cluster median as final beat.
function mbeats = fusion_mbeats(mbeats_c, aECG_c, fs)
    % Fusion of detected multichannel maternal ECG R-wave peak locations.
    % Reference: Qioung Yu, 2016.

    th2_1 = 0.6;
    th2_2 = 0.06*fs;
    th3_3 = 0.5;
    a = 2.5;

    [~, nb_c] = size(mbeats_c);
    ind = [];
    RRI_med_c = [];

    % Channel selection
    for i = 1:nb_c
        RRI = median(diff(mbeats_c{i}));
        if RRI < 0.4*fs || RRI > 1.6*fs
            continue
        end
        ind = [ind, i];
        RRI_med_c = [RRI_med_c, RRI];
    end
    nb_c = length(ind);
    RRI_m = median(RRI_med_c);
    MaximalQT = ceil(median(RRI_m)/2);
    med_diff_c = abs(RRI_med_c - RRI_m);
    mi = min(med_diff_c);
    error2_c = abs(med_diff_c - mi);

    score_c = [];
    for i = 1:nb_c
        sub_i = ind(i);
        mbeats = mbeats_c{sub_i};
        I = aECG_c{sub_i};
        RRI = diff(mbeats);
        RRI = [RRI(1) RRI];

        tmp = find(mbeats > MaximalQT);
        mbeats = mbeats(tmp);
        RRI = RRI(tmp);

        tmp = find(mbeats + MaximalQT <= length(I));
        mbeats = mbeats(tmp);
        RRI = RRI(tmp);

        V = [];
        II = [];
        for ii = 1:length(RRI)
            idx = mbeats(ii)-MaximalQT : mbeats(ii) + MaximalQT;
            V(:, ii) = I(idx);
            II = [II; idx];
        end

        template = median(V, 2);
        score1 = corrcoef([template, V]);
        score1 = score1(2:end, 1);
        score1 = sum(score1 >= th2_1)/length(score1);

        if error2_c(i) < th2_2
            score2 = 0.3;
        elseif (error2_c(i) < 2*th2_2) && (error2_c(i) >= th2_2)
            score2 = 0.2;
        else
            score2 = 0.1;
        end

        score_c = [score_c, score1*score2];
    end

    ma = max(score_c);
    ind = ind(score_c >= 0.5*ma);
    RRI_med_c = RRI_med_c(score_c >= 0.5*ma);
    RRI_m2 = median(RRI_med_c)';
    nb_c = length(ind);
    MaximalQT = ceil(median(RRI_m2)/2);

    % Second process
    for i = 1:nb_c
        sub_i = ind(i);
        mbeats = mbeats_c{sub_i};
        I = aECG_c{sub_i};
        RRI = diff(mbeats);
        RRI = [RRI(1) RRI];

        tmp = find(mbeats > MaximalQT);
        mbeats = mbeats(tmp);
        RRI = RRI(tmp);

        tmp = find(mbeats + MaximalQT <= length(I));
        mbeats = mbeats(tmp);
        RRI = RRI(tmp);

        V = [];
        II = [];
        for ii = 1:length(RRI)
            idx = mbeats(ii)-MaximalQT : mbeats(ii) + MaximalQT;
            V(:, ii) = I(idx);
            II = [II; idx];
        end

        template = median(V, 2);
        score3 = corrcoef([template, V]);
        score3 = score3(2:end, 1);
        mbeats_c{sub_i} = mbeats(score3 > th3_3);
    end

    X = [];
    for i = 1:nb_c
        X = [X, mbeats_c{ind(i)}];
    end

    if X
        R_peak = sort(X, 'ascend');
        Q = {};
        Q{1} = R_peak(1);
        k = 1;
        for j = 2:length(R_peak)
            if abs(R_peak(j) - R_peak(j-1)) <= 0.03*fs
                Q{k} = [Q{k}, R_peak(j)];
            else
                k = k + 1;
                Q{k} = R_peak(j);
            end
        end

        mbeats = [];
        for j = 1:k
            if length(Q{j}) >= nb_c/2
                mbeats = [mbeats, median(Q{j})];
            end
        end
        mbeats = round(mbeats);
    else
        mbeats = [];
    end

end
