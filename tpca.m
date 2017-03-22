function [Mproj,PC,latent] = tpca(M,d,varargin)
% TPCA Tensor principle component analysis (tPCA)
%
% [MPROJ, PC, LATENT] = TPCA(M,d) performs the tensor version of the
%   traditional PCA. Two versions are implemented. Given n tensor observations, flatten each along mode i
%   to obtain a matrix, take outer product of the matrix (p_i-by-p_i), then
%   do the classical PCA on the sum of outer products, retrieve the first
%   d_i principal components (p_i-by-d_i), then scale the original tensor
%   in the PC basis along each mode.
%
%   INPUT:
%       M: array variates (aka tensors) with dim(M) = [p_1,p_2,...,p_D,n]
%       d: target dimensions [d_1,...,d_D]
%
%   Optional input name-value pairs:
%       'Centering': true (default) | false, center tensor data or note
%       'Method': 'hosvd' (default) | '2dsvd', method for tensor PCA
%
%   Output:
%       MPROJ: tensor with dim(Mproj)=[d_1,...,d_D,n] after change of basis.
%           In PCA literature, it is called the SCOREs
%       PC: D-by-1 cell array containing principal components along each
%           mode. PC{i} has dimension p_i-by-d_i. In PCA literature, they
%           are called the COEFFICIENTs
%       LATENT: D-by-1 cell array containing the d_i eigen values along
%           each mode. Ordered from largest to smallest
%
% Examples
%
% See also pca
%
% Reference
%   L Lu, KN Plataniotis, and AN Venetsanopoulos (2006) Multilinear
%   principal component analysis for tensor objects for classification, 
%   Proc. Int. Conf. on Pattern Recognition.
%
% COPYRIGHT 2011-2013 North Carolina State University
% Hua Zhou <hua_zhou@ncsu.edu>

% parse inputs
argin = inputParser;
argin.addRequired('M', @(x) isa(x,'tensor') || isnumeric(x));
argin.addRequired('d', @isnumeric);
argin.addParamValue('Centering', true, @(x) islogical(x));
argin.addParamValue('Method', 'hosvd', @(x) ischar(x));
argin.parse(M,d,varargin{:});

centering = argin.Results.Centering;
method = argin.Results.Method;

% check dimensionalities
p = size(M); n = p(end); p(end) = []; D = length(p);
if size(d,1)>size(d,2)
    d = d';
end
if length(d)~=D
    error('tensorreg:tpca:wrongdim', ...
        'target dimensions do not match array dimension');
end
if any(d>p)
    error('tensorreg:tpca:exceeddim', ...
        'target dimensions cannot exceed original dimensions');
end

% change M to tensor (if it is not)
if isa(M,'tensor')
    TM = M;
else
    TM = tensor(M);
end

% centering data
idx = repmat({':'}, D, 1);
if centering

    TMavg = collapse(TM, D+1, @mean);
    for i=1:n
        TM(idx{:},i) = TM(idx{:},i) - TMavg;
    end

end

if strcmpi(method,'hosvd')
    
    warning('off','MATLAB:eigs:TooManyRequestedEigsForRealSym');
    tucker_approx = tucker_als(TM, [d n], 'printitn', false);
    warning on all;
    % PC basis
    PC = tucker_approx.U(1:D)';
    % tensor singular values
    latent = cell(1,D);
    for dd=1:D
        latent{dd} = double(collapse(tucker_approx.core, -dd, @norm));
        % sort in descending order
        [latent{dd}, IX] = sort(latent{dd}, 'descend');
        PC{dd} = PC{dd}(:,IX);
    end
    
elseif strcmpi(method,'2dsvd')

    % loop over dimensions to obtain PCs
    PC = cell(1,D);
    latent = cell(1,D);
    for dd=1:D
        C = zeros(p(dd),p(dd)); % p_d-by-p_d
        for i=1:n
            tmati = double(tenmat(TM(idx{:},i),dd));   % #rows = p_d
            C = C + tmati*tmati';
        end
        C = C/n;
        [PC{dd},latent{dd}] = eigs(C,d(dd));
        latent{dd} = diag(latent{dd});
        % sort in descending order
        [latent{dd}, IX] = sort(latent{dd}, 'descend');
        PC{dd} = PC{dd}(:,IX);
    end
    
end%if

% change of basis for original array data
%Mproj = ttm(TM,[cellfun(@(X) X',PC,'UniformOutput',false), eye(n)]);
PC = cellfun(@(X) X',PC,'UniformOutput',false);
PC{D+1} = speye(n);
Mproj = ttensor(TM, PC);
Mproj = double(Mproj);


end
