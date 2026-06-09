% PURPOSE:  Apply optimal hard-thresholding to the singular values of a noisy
%           low-rank matrix under the operator (spectral) norm loss. Derived from
%           the Gavish-Donoho (2014) asymptotically optimal shrinkage formula.
% INPUTS:   singvals — K×1 double, singular values from svd(data_matrix)
%           beta     — scalar ∈ (0,1], aspect ratio min(m,n)/max(m,n) of the data matrix
%           sigma    — scalar (optional), noise std per entry; estimated via
%                      Marchenko-Pastur median if omitted
% OUTPUTS:  singvals — K×1 double, shrunken singular values (small ones → 0)
% METHOD:   Operator-norm shrinkage: η(y) = max(x(y), 0) where
%           x(y) = √[0.5·((y²−β−1) + √((y²−β−1)²−4β))]·1(y≥1+√β)
function singvals = optimal_shrinkage(singvals, beta, sigma)
% Perform optimal shrinkage (operator norm loss, loss='op') on singular values.
%
% IN:
%   singvals: vector of data singular values from svd
%   beta:     aspect ratio m/n of the data matrix
%   sigma:    (optional) noise std; estimated from data if omitted
%
% OUT:
%   singvals: singular values after optimal shrinkage (op norm)
%
% Authors: Matan Gavish and David Donoho, 2013

    assert(prod(size(beta)) == 1)
    assert(beta <= 1)
    assert(beta > 0)
    assert(prod(size(singvals)) == length(singvals))

    if nargin < 3
        warning('off', 'MATLAB:quadl:MinStepSize')
        MPmedian = MedianMarcenkoPastur(beta);
        sigma = median(singvals) / sqrt(MPmedian);
    end

    singvals = optshrink_impl(singvals, beta, sigma);

end


function singvals = optshrink_impl(singvals, beta, sigma)
    assert(sigma > 0)
    assert(prod(size(sigma)) == 1)

    x = @(y)( sqrt(0.5*((y.^2 - beta - 1) + sqrt((y.^2 - beta - 1).^2 - 4*beta))) ...
        .* (y >= 1+sqrt(beta)));
    opt_op_shrink = @(y)( max(x(y), 0));

    singvals = sigma * opt_op_shrink(singvals/sigma);

end


function I = MarcenkoPasturIntegral(x, beta)
    if beta <= 0 || beta > 1
        error('beta beyond')
    end
    lobnd = (1 - sqrt(beta))^2;
    hibnd = (1 + sqrt(beta))^2;
    if (x < lobnd) || (x > hibnd)
        error('x beyond')
    end
    dens = @(t) sqrt((hibnd-t).*(t-lobnd)) ./ (2*pi*beta.*t);
    I = quadl(dens, lobnd, x);
    fprintf('x=%.3f,beta=%.3f,I=%.3f\n', x, beta, I);
end


function med = MedianMarcenkoPastur(beta)
    MarPas = @(x) 1 - incMarPas(x, beta, 0);
    lobnd = (1 - sqrt(beta))^2;
    hibnd = (1 + sqrt(beta))^2;
    change = 1;
    while change && (hibnd - lobnd > .001)
        change = 0;
        x = linspace(lobnd, hibnd, 5);
        for i = 1:length(x)
            y(i) = MarPas(x(i));
        end
        if any(y < 0.5)
            lobnd = max(x(y < 0.5));
            change = 1;
        end
        if any(y > 0.5)
            hibnd = min(x(y > 0.5));
            change = 1;
        end
    end
    med = (hibnd + lobnd) ./ 2;
end


function I = incMarPas(x0, beta, gamma)
    if beta > 1
        error('betaBeyond');
    end
    topSpec = (1 + sqrt(beta))^2;
    botSpec = (1 - sqrt(beta))^2;
    MarPas = @(x) IfElse((topSpec-x).*(x-botSpec) > 0, ...
        sqrt((topSpec-x).*(x-botSpec)) ./ (beta.*x) ./ (2.*pi), 0);
    if gamma ~= 0
        fun = @(x) (x.^gamma .* MarPas(x));
    else
        fun = @(x) MarPas(x);
    end
    I = quadl(fun, x0, topSpec);

    function y = IfElse(Q, point, counterPoint)
        y = point;
        if any(~Q)
            if length(counterPoint) == 1
                counterPoint = ones(size(Q)).*counterPoint;
            end
            y(~Q) = counterPoint(~Q);
        end
    end
end
