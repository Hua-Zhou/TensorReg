function [beta0_final,beta_final,glmstats_final,dev_final] = ...
    kruskal_reg(X,M,y,r,dist,varargin)
% KRUSKAL_REG Fit the rank-r Kruskal tensor regression
%   [BETA0,BETA,GLMSTATS,DEV] = KRUSKAL_REG(X,M,Y,R,DIST) fits the Kruskal
%   tensor regression using the regular covariate matrix X,
%   multidimensional array(or tensor) variates M, response Y, rank of the
%   Kruskal tensor regression R and the assumed distribution of the model
%   DIST. Available value of DIST is are 'normal', 'binomial', 'gamma',
%   'inverse gaussian' and 'poisson'. The result BETA0 is the regression
%   coefficient vector for the regular covariates matrix X, BETA is the
%   tensor regression coefficient of tensor covariates M, GLMSTATS is the
%   summary statistics of GLM fit from the last iteration and DEV is the
%   deviance of final model.
%
%   [BETA0,BETA,GLMSTATS,DEV] =
%   KRUSKAL_REG(X,M,Y,R,DIST,'PARAM1',val1,'PARAM2',val2...)allows you to
%   specify optional parameters to control the model fit. Available
%   parameter name/value pairs are:
%       'B0': starting point, it can be a numeric array or a tensor
%       'Display': 'off' (default) or 'iter'
%       'MaxIter': maximum iteration, default is 100
%       'Replicates': # of intitial points to try, default is 5
%       'TolFun': tolerence in objective value, default is 1e-4
%       'weights': observation weights, default is ones for each obs.
%
%   Input:
%       X: n-by-p0 regular covariate matrix
%       M: array variates (or tensors) with dim(M) = [p1,p2,...,pd,n]
%       y: n-by-1 respsonse vector
%       r: rank of Kruskal tensor regression
%       dist: 'normal'|'binomial'|'gamma'|'inverse gaussian'|'poisson'
%
%   Output:
%       beta0_final: regression coefficients for the regular covariates
%       beta_final: a tensor of regression coefficientsn for array variates
%       glmstats_final: GLM regression summary statistics from last iter.
%       dev_final: deviance of final model
%
% Examples
%
% See also kruskal_sparsereg, matrix_sparsereg, tucker_reg,
% tucker_sparsereg.
%
% TODO
%   - properly deal with the identifiability issue
%
% Reference
%   H Zhou, L Li, and H Zhu (2013) Tensor regression with applications in
%   neuroimaging data analysis, JASA, 108(502):540-552
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
argin.addParamValue('B0', [], @(x) isnumeric(x) || ...
    isa(x,'tensor') || isa(x,'ktensor') || isa(x,'ttensor'));
argin.addParamValue('Display', 'off', @(x) strcmp(x,'off') || ...
    strcmp(x,'iter'));
argin.addParamValue('MaxIter', 100, @(x) isnumeric(x) && x>0);
argin.addParamValue('TolFun', 1e-4, @(x) isnumeric(x) && x>0);
argin.addParamValue('Replicates', 5, @(x) isnumeric(x) && x>0);
argin.addParamValue('warn', false, @(x) islogical(x));
argin.addParamValue('weights', [], @(x) isnumeric(x) && all(x>=0));
argin.parse(X,M,y,r,dist,varargin{:});

B0 = argin.Results.B0;
Display = argin.Results.Display;
MaxIter = argin.Results.MaxIter;
TolFun = argin.Results.TolFun;
Replicates = argin.Results.Replicates;
warn = argin.Results.warn;
wts = argin.Results.weights;

% check validity of rank r
if isempty(r)
    r = 1;
elseif r==0
    [beta0_final,dev_final,glmstats_final] = ...
        glmfit_priv(X,y,dist,'constant','off','weights',wts);
    beta_final = 0;
    return;
end

% check dimensions
if isempty(X)
    X = ones(size(M,ndims(M)),1);
end
[n,p0] = size(X);
d = ndims(M)-1;             % dimension of array variates
p = size(M);                % sizes array variates
if p(end)~=n
    error('tensorreg:kruskal_reg:dim', ...
        'dimension of M does not match that of X!');
end
if n<p0 || n<r*max(p(1:end-1))
    error('tensorreg:kruskal_reg:smn', ...
        'sample size n is not large enough to estimate all parameters!');
end

% convert M into a tensor T (if it's not)
TM = tensor(M);

% if space allowing, pre-compute mode-d matricization of TM
if strcmpi(computer,'PCWIN64') || strcmpi(computer,'PCWIN32')
    iswindows = true;
    % memory function is only available on windows !!!
    [dummy,sys] = memory; %#ok<ASGLU>
else
    iswindows = false;
end
% CAUTION: may cause out of memory on Linux
if ~iswindows || d*(8*prod(size(TM)))<.75*sys.PhysicalMemory.Available %#ok<PSIZE>
    Md = cell(d,1);
    for dd=1:d
        Md{dd} = double(tenmat(TM,[d+1,dd],[1:dd-1 dd+1:d]));
    end
end

% check user-supplied initial point
if ~isempty(B0)
    % ignore requested multiple initial points
    Replicates = 1;
    % check dimension
    if ndims(B0)~=d
        error('tensorreg:kruskal_reg:badB0', ...
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
    % perform CP decomposition if it's not a ktensor of correct rank
    if isa(B0,'tensor') || isa(B0,'ttensor') || ...
            (isa(B0, 'ktensor') && size(B0.U{1},2)~=r)
        B0 = cp_als(B0, r, 'printitn', 0);
    end
end

% turn off warnings from glmfit_priv
if ~warn
    warning('off','stats:glmfit:IterationLimit');
    warning('off','stats:glmfit:BadScaling');
    warning('off','stats:glmfit:IllConditioned');
end

% pre-allocate variables
glmstats = cell(1,d+1);
dev_final = inf;

% loop for various intial points
for rep=1:Replicates
    
    if ~isempty(B0)
        beta = B0;
    else
        % initialize tensor regression coefficients from uniform [-1,1]
        beta = ktensor(arrayfun(@(j) 1-2*rand(p(j),r), 1:d, ...
            'UniformOutput',false));
    end
    
    % main loop
    for iter=1:MaxIter
        % update coefficients for the regular covariates
        if (iter==1)
            [beta0, dev0] = glmfit_priv(X,y,dist,'constant','off', ...
                'weights',wts);
        else
            eta = Xj*beta{d}(:);
            [betatmp,devtmp,glmstats{d+1}] = ...
                glmfit_priv([X,eta],y,dist,'constant','off','weights',wts);
            beta0 = betatmp(1:end-1);
            % stopping rule
            diffdev = devtmp-dev0;
            dev0 = devtmp;
            if (abs(diffdev)<TolFun*(abs(dev0)+1))
                break;
            end
            % update scale of array coefficients and standardize
            beta = arrange(beta*betatmp(end));
            for j=1:d
                beta.U{j} = bsxfun(@times,beta.U{j},(beta.lambda').^(1/d));
            end
            beta.lambda = ones(r,1);            
        end
        % cyclic update of the array coefficients
        eta0 = X*beta0;
        for j=1:d
            if j==1
                cumkr = ones(1,r);
            end
            if (exist('Md','var'))
                if j==d
                    Xj = reshape(Md{j}*cumkr,n,p(j)*r);
                else
                    Xj = reshape(Md{j}*khatrirao([beta.U(d:-1:j+1),cumkr]),...
                        n,p(j)*r);
                end
            else
                if j==d
                    Xj = reshape(double(tenmat(TM,[d+1,j]))*cumkr, ...
                        n,p(j)*r);
                else
                    Xj = reshape(double(tenmat(TM,[d+1,j])) ...
                        *khatrirao({beta.U{d:-1:j+1},cumkr}),n,p(j)*r);
                end
            end
            [betatmp,dummy,glmstats{j}] = ...
                glmfit_priv([Xj,eta0],y,dist,'constant','off', ...
                'weights',wts); %#ok<ASGLU>
            beta{j} = reshape(betatmp(1:end-1),p(j),r);
            eta0 = eta0*betatmp(end);
            cumkr = khatrirao(beta{j},cumkr);
        end
    end
    
    % record if it has a smaller deviance
    if dev0<dev_final
        beta0_final = beta0;
        beta_final = beta;
        glmstats_final = glmstats;
        dev_final = dev0;
    end
    
    if strcmpi(Display,'iter')
        disp(' ');
        disp(['replicate: ' num2str(rep)]);
        disp([' iterates: ' num2str(iter)]);
        disp([' deviance: ' num2str(dev0)]);
        disp(['    beta0: ' num2str(beta0')]);
    end
    
end

% turn warnings on
if ~warn
    warning('on','stats:glmfit:IterationLimit');
    warning('on','stats:glmfit:BadScaling');
    warning('on','stats:glmfit:IllConditioned');
end

% output BIC of the final model. Note deviance = -2*log-likelihood
if d==2
    glmstats_final{d+1}.BIC = dev_final + log(n)*(r*(p(1)+p(2)-r)+p0);
else
    glmstats_final{d+1}.BIC = dev_final + ...
        log(n)*(r*(sum(p(1:end-1))-d+1)+p0);
end
glmstats_final{d+1}.yhat = y - glmstats_final{d+1}.resid;

end