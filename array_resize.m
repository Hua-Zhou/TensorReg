function [B] = array_resize(A, targetdim, varargin)
% ARRAY_RESIZE Resize a multi-dimensional array A
%
% B = ARRAY_RESIZE(A, TARGETDIM) resizes an array A to target dimension.
% When 'method' equals 'interpolate' or 'dct', both downsizing and upsizing
% are allowed. When 'method' equals 'hosvd' or '2dsvd', only downsizing is
% allowed and the last element of targetdim is ignored. The resized array B
% has same data type as A.
%
%   INPUT
%       A: a D-dimensional array or tensor or ktensor or ttensor
%       targetdim: a 1-by-D vector of target dimensions
%
%   Optional input name-value pairs:
%       'method': 'interpolate'(default) | 'dct' | 'hosvd' | '2dsvd'
%
%   OUTPUT
%       B: resized array, of same type as A
%
% Examples
%
% See also imresize
%
% TODO
%   - wavelet transform
%   - methods 'hosvd' and '2dsvd' don't work for tensor input yet
%
% Copyright 2011-2013 North Carolina State University
% Hua Zhou <hua_zhou@ncsu.edu>

% parse inputs
argin = inputParser;
argin.addRequired('A');
argin.addRequired('targetdim', @isnumeric);
argin.addParamValue('method', 'interpolate', @(x) ischar(x));
argin.addParamValue('inverse', false, @islogical);
argin.parse(A, targetdim, varargin{:});

isinverse = argin.Results.inverse;
method = argin.Results.method;

% turn array into tensor structure
p = size(A);
D = length(p);

% argument checking
if length(targetdim)~=D
    error('tensorreg:array_resize:targetdim', ...
        'targetdim does not match dimension of A');
end

switch method

    case 'interpolate'  % interpolation method
        
        U = cell(1,D);
        for d=1:D
            xi = linspace(1, p(d), targetdim(d));
            j1 = floor(xi);
            j2 = j1 + 1;
            w1 = j2 - xi;
            w2 = 1 - w1;
            j2(end) = p(d);
            % the interpolation matrix: targetdim(d)-by-p(d)
            U{d} = sparse([1:targetdim(d) 1:targetdim(d)], [j1 j2], ...
                [w1 w2], targetdim(d), p(d));
        end
        if isa(A, 'ktensor') || isa(A, 'ttensor')
            B = A;
            for d=1:D
                B.U{d} = U{d}*B.U{d}; % resize factor matrix
            end
        elseif isa(A, 'tensor') % tensor or array
            B = tensor(ttensor(tensor(A),U));
        else
            B = double(ttensor(tensor(A),U));
            % cast back to the previous class
            B = cast(B, class(A));
        end

    case 'dct'  % discrete cosine transform
        
        if isa(A, 'ktensor') || isa(A, 'ttensor')
            B = A';
            for d=1:D
                B.U{d} = resize(B.U{d}, [targetdim(d), size(B.U{d},2)]);
            end
        elseif isa(A, 'tensor')
            B = tensor(resize(double(A), targetdim));
        else
            B = resize(A, targetdim);
        end
        
    case 'hosvd' % tensor PCA via HOSVD
        
        B = tpca(double(A), targetdim(1:end-1), 'method', 'hosvd');
        B = cast(B, class(A));
        
    case '2dsvd' % tensor PCA via marginal SVD
    
        B = tpca(double(A), targetdim(1:end-1), 'method', '2dsvd');
        B = cast(B, class(A));
        
    otherwise   % wavelet basis
        
        % obtain the support of the wavelets
        [hr] = wfilters(method,'r');
        N = fix(length(hr)/2);
        
        % U holds the component matrices in Tucker tensor
        U = cell(1,D);
        for d=1:D
            if isinverse
                Kd = wpfun(method,p(d)-1,ceil(log2(targetdim(d)/(2*N-1))));
                Kd = bsxfun(@times, Kd, 1./sqrt(sum(Kd.^2,2)));
                U{d} = array_resize(Kd,[p(d),targetdim(d)])';
            else
                Kd = wpfun(method,targetdim(d)-1,ceil(log2(p(d)/(2*N-1))));
                Kd = bsxfun(@times, Kd, 1./sqrt(sum(Kd.^2,2)));
                U{d} = array_resize(Kd,[targetdim(d),p(d)]);
            end
        end
        % cast back to the previous class
        B = cast(double(ttensor(tensor(A),U)), class(A));

end%switch

end