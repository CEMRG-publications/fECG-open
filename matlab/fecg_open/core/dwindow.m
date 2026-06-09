function Dh = dwindow(h)
% dwindow  Compute the derivative of an analysis window.
%
% PURPOSE
%   Returns the time-derivative of a symmetric analysis window h using a
%   centred finite-difference scheme.  The derivative is needed to compute
%   the instantaneous frequency deviation (IFD) term in the STFT-based
%   synchrosqueezing transform (see STFT_IFD_fast.m).
%
% INPUTS
%   h  - (L x 1 real) analysis window column vector of odd length L = 2*Lh+1.
%
% OUTPUTS
%   Dh - (L x 1 real) derivative of h, same length and orientation as h.
%
%   Adapted from the SST_compare toolbox by H. Yang (BSD-3-Clause):
%   https://github.com/HaizhaoYang/SST_compare
%   Original author: F. Auger. Copyright (c) 1996 CNRS (France).

    if nargin == 0, error("one parameter required"); end
    [hrow, hcol] = size(h);
    if hcol ~= 1, error("h must have only one column"); end
    Lh = (hrow - 1) / 2;
    step_height = (h(1) + h(hrow)) / 2;
    ramp = (h(hrow) - h(1)) / (hrow - 1);
    h2 = [0; h - step_height - ramp*(-Lh:Lh).'; 0];
    Dh = (h2(3:hrow+2) - h2(1:hrow)) / 2 + ramp;
    Dh(1)    = Dh(1)    + step_height;
    Dh(hrow) = Dh(hrow) - step_height;

end
