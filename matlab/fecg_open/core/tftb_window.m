function h = tftb_window(N)
% tftb_window  Generate a flat-top analysis window of length N.
%
% PURPOSE
%   Produces a 3-term flat-top (Blackman-Harris family) window that
%   minimises spectral leakage.  Used as the STFT analysis window in the
%   synchrosqueezing transform pipeline (CFPH.m, STFT_IFD_fast.m).
%
% INPUTS
%   N  - (scalar positive integer) window length in samples.
%
% OUTPUTS
%   h  - (N x 1 real) flat-top window coefficients.
%
%   Adapted from the SST_compare toolbox by H. Yang (BSD-3-Clause):
%   https://github.com/HaizhaoYang/SST_compare
%   Original author: F. Auger. Copyright (c) 1996 CNRS (France).

    if nargin == 0, error("at least 1 parameter is required"); end
    if N <= 0, error("N must be strictly positive"); end
    ind = (-(N-1)/2 : (N-1)/2)' * 2*pi / (N-1);
    h = 0.2810639 + 0.5208972*cos(ind) + 0.1980399*cos(2*ind);

end
