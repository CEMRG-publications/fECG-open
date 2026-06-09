% PURPOSE:  Isolate the periodic (beat-aligned) ECG component using SVD-based
%           optimal shrinkage. Used both to extract the mECG template and to
%           subtract it, leaving the fECG residual.
% INPUTS:   x0            — 1×N double, de-trended mixed ECG (for beat timing)
%           x0_real       — 1×N double, morphology-preserving mixed ECG
%           current_beats — 1×K integer, R-peak sample indices
%           sigma_coeff   — scalar, noise-level multiplier (1.5 in main)
%           ifauto        — 0/1 flag; 0 = use sigma=1 (fixed), 1 = estimate from data
% OUTPUTS:  Om0      — 1×N double, reconstructed periodic ECG component (timing version)
%           Om0_real — 1×N double, reconstructed periodic ECG component (morph version)
% METHOD:   Build a beat matrix V (beat-window × beat-count), apply operator-norm
%           optimal shrinkage to V's singular values, reconstruct with overlap-add
%           weighting at beat boundaries. Window half-width = 95th percentile of RRI × 0.5.
function [Om0, Om0_real] = ECG_shrinkage0(x0, x0_real, current_beats, sigma_coeff, ifauto)
    % optimal shrinkage of ECG
    RRI = diff(current_beats);
    RRI = [RRI(1) RRI];
    MaximalQTp = ceil(quantile(RRI, 0.95)*4/8);
    MaximalQTt = ceil(quantile(RRI, 0.95)*4/8);

    tmp = find(current_beats > MaximalQTp);
    current_beats = current_beats(tmp);
    RRI = RRI(tmp);

    tmp = find(current_beats + MaximalQTt <= length(x0));
    current_beats = current_beats(tmp);
    RRI = RRI(tmp);

    V = [];
    V2 = [];
    II = [];
    count = 0;
    for ii = 1:length(current_beats)
        count = count + 1;
        idx = (current_beats(ii)-MaximalQTp) : (current_beats(ii) + MaximalQTt);
        V(:, count) = x0(idx);
        V2(:, count) = x0_real(idx);
        II = [II; idx];
    end

    [n_t, n_theta] = size(V);

    sigma = (V - (median(V, 2)*(ones(1, size(V, 2))))).^2;
    sigma = sigma_coeff*sqrt(sum(sum(sigma)/n_t)/n_theta);
    [n_t, n_theta] = size(V);

    if n_theta > n_t
        beta0 = n_t/n_theta;
        [ae, be, ce] = svd(V./(sigma*sqrt(n_theta)));
        lambdaOSe = diag(be);
        if ifauto
            singvals = optimal_shrinkage(lambdaOSe, beta0);
        else
            singvals = optimal_shrinkage(lambdaOSe, beta0, 1);
        end
        XN0 = sigma*sqrt(n_theta).*(ae*diag(singvals)*ce(:, 1:n_t)');
    else
        beta0 = n_theta/n_t;
        [ae, be, ce] = svd(V'./(sigma*sqrt(n_t)));
        lambdaOSe = diag(be);
        if ifauto
            singvals = optimal_shrinkage(lambdaOSe, beta0);
        else
            singvals = optimal_shrinkage(lambdaOSe, beta0, 1);
        end
        XN0 = sigma*sqrt(n_t).*(ae*diag(singvals)*ce(:, 1:n_theta)')';
    end

    sigma = (V2 - (median(V2, 2)*(ones(1, size(V2, 2))))).^2;
    sigma = sigma_coeff*sqrt(sum(sum(sigma)/n_t)/n_theta);
    if n_theta > n_t
        beta0 = n_t/n_theta;
        [ae, be, ce] = svd(V2./(sigma*sqrt(n_theta)));
        lambdaOSe = diag(be);
        if ifauto
            singvals = optimal_shrinkage(lambdaOSe, beta0);
        else
            singvals = optimal_shrinkage(lambdaOSe, beta0, 1);
        end
        XN02 = sigma*sqrt(n_theta).*(ae*diag(singvals)*ce(:, 1:n_t)');
    else
        beta0 = n_theta/n_t;
        [ae, be, ce] = svd(V2'./(sigma*sqrt(n_t)));
        lambdaOSe = diag(be);
        if ifauto
            singvals = optimal_shrinkage(lambdaOSe, beta0);
        else
            singvals = optimal_shrinkage(lambdaOSe, beta0, 1);
        end
        XN02 = sigma*sqrt(n_t).*(ae*diag(singvals)*ce(:, 1:n_theta)')';
    end

    Om0 = zeros(1, length(x0));
    Om0_real = zeros(1, length(x0_real));

    for s = 1:length(current_beats)

        XN_to_add = XN0(:, s);
        XN2_to_add = XN02(:, s);

        % reconstruction
        if s == 1
            left_overlap = 2;
            right_overlap = length(intersect(II(s+1, :), II(s, :)));
        elseif s == length(current_beats)
            left_overlap = length(intersect(II(s-1, :), II(s, :)));
            right_overlap = 2;
        else
            left_overlap = length(intersect(II(s-1, :), II(s, :)));
            right_overlap = length(intersect(II(s+1, :), II(s, :)));
        end

        if left_overlap <= 1; left_overlap = 2; end
        if right_overlap <= 1; right_overlap = 2; end

        W = ones(MaximalQTp+MaximalQTt+1, 1);
        W(1:left_overlap) = sin(linspace(0, pi/2, left_overlap)).^2;
        W(end:-1:end-right_overlap+1) = sin(linspace(0, pi/2, right_overlap)).^2;

        Om_toadd = (W.*XN_to_add)';
        Om_real_toadd = (W.*XN2_to_add)';

        Om0(II(s, :)) = Om0(II(s, :)) + Om_toadd;
        Om0_real(II(s, :)) = Om0_real(II(s, :)) + Om_real_toadd;
    end

    Om0(Om0 == 0) = x0(Om0 == 0);
    Om0_real(Om0_real == 0) = x0_real(Om0_real == 0);

end
