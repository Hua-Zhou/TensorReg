function [beta0_final,beta_final,beta_scale,glmstats] = ...
    tucker_sparsereg(X,M,y,r,dist,lambda,pentype,penparam,varargin)
% TUCKER_SPARSEREG Fit the rank-r GLM sparse Tucker tensor regression
%   [BETA0,BETA,BETA_SCALE,GLMSTATS] =
%   TUCKER_SPARSEREG(X,M,Y,R,DIST,LAMBDA,PENTYPE,PENPARAM) fits the sparse
%   Tucker tensor regression using the regular covariate matrix X,
%   multidimensional array(or tensor) variates M, response Y, rank of the
%   Tucker tensor regression R, the assumed distribution of the model DIST,
%   the penalty PENTYPE at fixed tuning parameter value LAMBDA, and
%   the index of the penalty type PENPARAM. Available value of DIST is are
%   'normal', 'binomial', 'gamma', 'inverse gaussian' and 'poisson'. For
%   the input value PENTYPE, available penalties are 'enet',
%   'log','mcp','power' and 'scad'. The result BETA0 is the regression
%   coeffiient vector for the regular covariates matrix X, BETA is the
%   tensor regression coefficient of the tensor covariates M, BETA_SCALE is
%   the tensr of the scaling constants for the tensor covariates M,
%   GLMSTATS is the summary statistics of GLM fit from the last iteration.
%
%   [BETA0,BETA,BETA_SCALE,GLMSTATS] =
%   TUCKER_SPARSEREG(X,M,Y,R,DIST,LAMBDA,PENTYPE,PENPARAM,'PARAM1',val1,
%   'PARAM2',val2,...) allows you to specify optional parameters to control
%   the model fit. Availavle parameter name/value pairs are :
%       'B0': starting point, it can be a numeric array or a tensor
%       'Display': 'off' (default) or 'iter'
%       'BurninMaxIter': Max. iter. for the burn-in runs, default is 20
%       'BurninTolFun': Tolerance for the burn-in runs, default is 1e-2
%       'BurninReplicates': Number of the burn-in runs, default is 10
%       'PenaltyMaxIter': Max. iters. at penalization stage, default is 50
%       'PenaltyTolFun': Tolerence at penalization stage, default is 1e-3
%       'weights': Observation weights, default is ones for each obs.
%
%   INPUT:
%       X: n-by-p0 regular covariate matrix
%       M: array variates (or tensors) with dim(M) = [p1,p2,...,pd,n]
%       y: n-by-1 respsonse vector
%       r: rank of tensor regression
%       dist: 'binomial', 'normal', or 'poisson'
%       lambda: penalty tuning constant
%       pentype: 'enet'|'log'|'mcp'|'power'|'scad'
%       penparam: the index parameter for the pentype
%
%   Output:
%       beta0_final: regression coefficients for the regular covariates
%       beta_final: a tensor of regression coefficientsn for array variates
%       beta_scale: a tensor of the scaling constants for the array
%           coefficients
%       glmstats: GLM statistics from the last fitting of the regular
%           covariates
%
% Examples
%
% See also kruskal_reg, kruskal_sparsereg, matrix_sparsereg, tucker_reg.
%
% Reference
%   X Li, H Zhou, L Li (2013) Tucker Tensor Regression and Neuroimaging
%   Analysis, submitted. <http://arxiv.org/abs/1304.5637>
%
% COPYRIGHT 2011-2013 North Carolina State University
% Hua Zhou <hua_zhou@ncsu.edu>

% parse inputs
argin = inputParser;
argin.addRequired('X', @isnumeric);
argin.addRequired('M', @(x) isa(x,'tensor') || isnumeric(x));
argin.addRequired('y', @isnumeric);
argin.addRequired('r', @isnumeric);
argin.addRequired('dist', @(x) ischar(x));
argin.addRequired('lambda', @(x) isnumeric(x) && x>=0);
argin.addRequired('pentype', @ischar);
argin.addRequired('penparam', @isnumeric);
argin.addParamValue('B0', [], @(x) isnumeric(x) || ...
    isa(x,'tensor') || isa(x,'ktensor') || isa(x,'ttensor'));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off')||strcmp(x,'iter'));
argin.addParamValue('BurninMaxIter', 20, @(x) isnumeric(x) && x>0);
argin.addParamValue('BurninTolFun', 1e-2, @(x) isnumeric(x) && x>0);
argin.addParamValue('BurninReplicates', 10, @(x) isnumeric(x) && x>0);
argin.addParamValue('PenaltyMaxIter', 50, @(x) isnumeric(x) && x>0);
argin.addParamValue('PenaltyTolFun', 1e-3, @(x) isnumeric(x) && x>0);
argin.addParamValue('warn', false, @(x) islogical(x));
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(X,M,y,r,dist,lambda,pentype,penparam,varargin{:});

B0 = argin.Results.B0;
Display = argin.Results.Display;
BurninMaxIter = argin.Results.BurninMaxIter;
BurninTolFun = argin.Results.BurninTolFun;
BurninReplicates = argin.Results.BurninReplicates;
PenaltyMaxIter = argin.Results.BurninMaxIter;
PenaltyTolFun = argin.Results.PenaltyTolFun;
warn = argin.Results.warn;
wts = argin.Results.weights;
if (isempty(wts))
    wts = ones(size(X,1),1);
end

% check positivity of tuning parameter
if lambda==0
    error('tensorreg:tucker_sparsereg:nopen', ...
        'lambda=0 (no penalization); call tucker_reg instead');    
end

% check validity of rank r
if (isempty(r))
    error('tensorreg:tucker_sparsereg:norank', ...
        'need to input the d-by-1 rank vector!');
elseif (any(r==0))
    [beta0_final,dev_final,glmstats] = ...
        glmfit_priv(X,y,dist,'constant','off','weights',wts); %#ok<ASGLU>
    beta_final = 0;
    return;
elseif (size(r,1)==1 && size(r,2)==1)
    r = repmat(r,1,ndims(M)-1);
end

% decide least squares or GLM model
if (strcmpi(dist,'normal'))
    isglm = false;
else
    isglm = true;
    % translate to model specifier for sparse regression
    if (strcmpi(dist,'binomial'))
        glmmodel = 'logistic';
    elseif (strcmpi(dist,'poisson'))
        glmmodel = 'loglinear';
    end
end

% check dimensions
[n,p0] = size(X);
d = ndims(M)-1; % dimension of array variates
p = size(M);    % sizes array variates
if (n~=p(end))
    error('tensorreg:tucker_sparsereg:nmismatch', ...
        'sample size in X dose not match sample size in M');
end
if (n<p0 || n<max(r.*p(1:end-1)))    
    error('tensorreg:tucker_sparsereg:smalln', ...
        'sample size n is not large enough to estimate all parameters!');
end

% convert M into a tensor T
TM = tensor(M);

% if space allowing, pre-compute mode-d matricization of TM
if (strcmpi(computer,'PCWIN64') || strcmpi(computer,'PCWIN32'))
    iswindows = true;
    % memory function is only available on windows !!!
    [dummy,sys] = memory; %#ok<ASGLU>
else
    iswindows = false;
end
% CAUTION: may cause out of memory on Linux
if (~iswindows || d*(8*prod(size(TM)))<.75*sys.PhysicalMemory.Available) %#ok<PSIZE>
    Md = cell(d,1);
    for dd=1:d
        Md{dd} = double(tenmat(TM,[d+1,dd],[1:dd-1 dd+1:d]));
    end
end
Mn = double(tenmat(TM,d+1,1:d));    % n-by-prod(p)

% Burn-in stage (loose convergence criterion)
if (~strcmpi(Display,'off'))
    display(' ');
    display('==================');
    display('Burn-in stage ...');
    display('==================');
end

% reduce tensor size for reliable estimation in burnin stage
if isempty(B0) % no user-supplied start point
    if strcmpi(dist,'normal')
        shrink_factor = (n/5) / (sum(p(1:end-1).*r)+prod(r));
    elseif strcmpi(dist,'binomial')
        shrink_factor = (n/10) / (sum(p(1:end-1).*r)+prod(r));
    elseif strcmpi(dist,'poisson')
        shrink_factor = (n/10) / (sum(p(1:end-1).*r)+prod(r));
    end
    if shrink_factor <=1
        [dummy,beta_burnin] = tucker_reg(X,M,y,r,dist, ...
            'MaxIter',BurninMaxIter, ...
            'TolFun',BurninTolFun,...
            'Replicates',BurninReplicates,'weights',wts); %#ok<ASGLU>
    else
        targetdim = round(p(1:end-1)/shrink_factor);
        M_reduce = array_resize(M, targetdim);
        % estimate at reduced dimension
        [dummy,beta_burnin] = tucker_reg(X,M_reduce,y,r,dist, ...
            'MaxIter',BurninMaxIter, ...
            'TolFun',BurninTolFun,...
            'Replicates',BurninReplicates,'weights',wts); %#ok<ASGLU>
        % resize back to original dimension
        beta_burnin = array_resize(beta_burnin, p(1:end-1));
        % warm start from coarsened estimate
        [dummy,beta_burnin] = tucker_reg(X,M,y,r,dist, ...
            'B0', beta_burnin, ...
            'MaxIter',BurninMaxIter, ...
            'TolFun',BurninTolFun,...
            'weights',wts); %#ok<ASGLU>
    end
else % user-supplied start point
    % check dimension
    if ndims(B0)~=d
        error('tensorreg:tucker_sparsereg:badB0', ...
            'dimension of B0 does not match that of data!');
    end
    % turn B0 into a tensor (if it's not)
    if isnumeric(B0)
        B0 = tensor(B0);
    end
    % resize to compatible dimension (if it's not)
    if any(size(B0)~=p(1:end-1))
        B0 = array_resize(B0, p);
    end
    % perform Tucker decomposition if it's not a Tucker tensor of correct rank
    if isa(B0,'tensor') || isa(B0,'ktensor') || ...
            (isa(B0, 'ttensor') && any(size(B0.core)~=r))
        B0 = tucker_als(B0, r, 'printitn', 0);
    end
    beta_burnin = B0;
end

% penalization stage
if ~strcmpi(Display,'off')
    display(' ');
    display('==================');
    display('Penalization stage');
    display('==================');
end
% turn off warnings
if ~warn
    warning('off','stats:glmfit:IterationLimit');
    warning('off','stats:glmfit:BadScaling');
    warning('off','stats:glmfit:IllConditioned');
end
glmstats = cell(1,d+2);
dev0 = inf;
beta = beta_burnin;
for iter = 1:PenaltyMaxIter
    
    % update coefficients for the regular covariates
    if (iter==1)
        eta = double(tenmat(TM,d+1)*tenmat(beta,1:d));
    else
        eta = Xcore*betatmp(1:end-1);
    end
    [betatmp,devtmp,glmstats{d+2}] = glmfit_priv([X,eta],y,dist, ...
        'constant','off','weights',wts);
    beta0 = betatmp(1:p0);
    % stopping rule
    diffdev = devtmp-dev0;
    dev0 = devtmp;
    if (abs(diffdev)<PenaltyTolFun*(abs(dev0)+1))
        break;
    end
    % update scale of array coefficients and standardize
    for j=1:d
        colnorms = sqrt(sum(beta.U{j}.^2,1));
        colnorms(colnorms==0) = 1;
        beta.U{j} = bsxfun(@times,beta.U{j},1./colnorms);
        beta.core = ttm(beta.core,diag(colnorms),j);
    end
    beta.core = beta.core*betatmp(end);
    % cyclic update of the array coefficients
    eta0 = X*beta0;
    for j=1:d
        if (j==1)
            cumkron = 1;
        end
        if (iter>1)
        if (exist('Md','var'))
            % need to optimize the computation!
            if (j==d)
                Xj = reshape(Md{j}*cumkron...
                    *double(tenmat(beta.core,j))', n, p(j)*r(j));
            else
                Xj = reshape(Md{j}...
                    *arraykron([beta.U(d:-1:j+1),cumkron])...
                    *double(tenmat(beta.core,j))', n, p(j)*r(j));
            end
        else
            if (j==d)
                Xj = reshape(double(tenmat(TM,[d+1,j]))*cumkron...
                    *double(tenmat(beta.core,j))', n, p(j)*r(j));
            else
                Xj = reshape(double(tenmat(TM,[d+1,j])) ...
                    *arraykron([beta.U(d:-1:j+1),cumkron])...
                    *double(tenmat(beta.core,j))', n, p(j)*r(j));
            end
        end
        [betatmp,devtmp,glmstats{j}] = ...
            glmfit_priv([Xj,eta0],y,dist,'constant','off','weights',wts); %#ok<ASGLU>
        beta{j} = reshape(betatmp(1:end-1),p(j),r(j));
        eta0 = eta0*betatmp(end);
        end
        cumkron = kron(beta{j},cumkron);
    end
    % update the core tensor
    Xcore = Mn*cumkron; % n-by-prod(r)
    if (isglm)
        betatmp = glm_sparsereg([Xcore,eta0],y,lambda,glmmodel,'weights',wts,...
            'x0',[beta.core(:);0],'penidx',[true(1,prod(r)),false],...
            'penalty',pentype,'penparam',penparam);
    else
        betatmp = lsq_sparsereg([Xcore,eta0],y,lambda,'weights',wts,...
            'x0',[beta.core(:);0],'penidx',[true(1,prod(r)),false],...
            'penalty',pentype,'penparam',penparam);
    end
    beta.core = tensor(betatmp(1:end-1),r);
   
    if (~strcmpi(Display,'off'))
        disp(' ');
        disp(['  iterate: ' num2str(iter)]);
        disp([' deviance: ' num2str(dev0)]);
        disp(['    beta0: ' num2str(beta0')]);
    end
end
beta0_final = beta0;
beta_final = beta;

% turn off warnings
if ~strcmpi(Display,'off')
    display(' ');
    display('==================');
    display('Scaling stage');
    display('==================');
end

% find a scaling for the estimates
beta_scale = ttensor(rand(r), arrayfun(@(j) zeros(p(j),r(j)), 1:d, ...
    'UniformOutput',false));
eta0 = X*beta0;
for j=1:d
    if (j==1)
        cumkron = 1;
    end
    if (exist('Md','var'))
        % need to optimize the computation!
        if (j==d)
            Xj = reshape(Md{j}*cumkron...
                *double(tenmat(beta.core,j))', n, p(j)*r(j));
        else
            Xj = reshape(Md{j}...
                *arraykron([beta.U(d:-1:j+1),cumkron])...
                *double(tenmat(beta.core,j))', n, p(j)*r(j));
        end
    else
        if (j==d)
            Xj = reshape(double(tenmat(TM,[d+1,j]))*cumkron...
                *double(tenmat(beta.core,j))', n, p(j)*r(j));
        else
            Xj = reshape(double(tenmat(TM,[d+1,j])) ...
                *arraykron([beta.U(d:-1:j+1),cumkron])...
                *double(tenmat(beta.core,j))', n, p(j)*r(j));
        end
    end
    [betatmp,devtmp,glmstats{j}] = ...
        glmfit_priv([Xj,eta0],y,dist,'constant','off','weights',wts); %#ok<ASGLU>
    beta_scale{j} = reshape(glmstats{j}.se(1:end-1),p(j),r(j));    
    cumkron = kron(beta{j},cumkron);
end
Xcore = Mn*cumkron; % n-by-prod(r)
[betatmp,devtmp,glmstats{d+1}] = ...
    glmfit_priv([Xcore,eta0],y,dist,'constant','off','weights',wts); %#ok<ASGLU>
beta_scale.core = reshape(glmstats{d+1}.se(1:end-1),r);

% output the BIC
glmstats{d+2}.BIC  = dev0 + log(n)*...
    (sum(arrayfun(@(dd) nnz(collapse(tensor(beta.core),-dd,@nnz)), 1:d).*p(1:d)) ...
    + nnz(beta.core) + p0);

% say goodbye
if (~strcmpi(Display,'off'))
    disp(' ');
    disp(' DONE!');
    disp(' ');
end

% turn warnings on
if ~warn
    warning('on','stats:glmfit:IterationLimit');
    warning('on','stats:glmfit:BadScaling');
    warning('on','stats:glmfit:IllConditioned');
end

    function X = arraykron(U)
        %ARRAYKRON Kronecker product of matrices in an array
        %   AUTHOR: Hua Zhou (hua_zhou@ncsu.edu)
        X = U{1};
        for i=2:length(U)
            X = kron(X,U{i});
        end
    end

    function X = kron(A,B)
        %KRON Kronecker product.
        %   kron(A,B) returns the Kronecker product of two matrices A and B, of
        %   dimensions I-by-J and K-by-L respectively. The result is an I*J-by-K*L
        %   block matrix in which the (i,j)-th block is defined as A(i,j)*B.
        
        %   Version: 03/10/10
        %   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
        
        [I, J] = size(A);
        [K, L] = size(B);
        if ~issparse(A) && ~issparse(B)
            A = reshape(A,[1 I 1 J]);
            B = reshape(B,[K 1 L 1]);
            X = reshape(bsxfun(@times,A,B),[I*K J*L]);
        else
            [ia,ja,sa] = find(A);
            [ib,jb,sb] = find(B);
            ix = bsxfun(@plus,K*(ia-1).',ib);
            jx = bsxfun(@plus,L*(ja-1).',jb);
            if islogical(sa) && islogical(sb)
                X = sparse(ix,jx,bsxfun(@and,sb,sa.'),I*K,J*L);
            else
                X = sparse(ix,jx,double(sb)*double(sa.'),I*K,J*L);
            end
        end
    end

end