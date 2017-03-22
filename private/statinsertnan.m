function [varargout]=statinsertnan(wasnan,varargin)
%STATINSERTNAN Insert NaN, space, '' or undefined value into inputs.
%   X1 = STATINSERTNAN(WASNAN, Y1) inserts missing values in Y1 and returns
%   it as X1. WASNAN is a logical column vector and the output of
%   STATREMOVENAN. Its TRUE values indicate the rows in X1 that will
%   contain missing values. Y1 is a column vector or a matrix. The type of
%   Y1 can be:
%   Categorical       - X1 is categorical, undefined values represents
%                       missing values.
%   Double            - X1 is double. NaN values represents missing values.
%   Single            - X1 is single. NaN values represents missing values.
%   Character matrix  - X1 is a character matrix. Space represents missing
%                       values.
%   Cell              - X1 is a cell array. empty string '' represents
%                       missing values.
%
%  [X1,X2,...] = STATINSERTNAN(WASNAN,Y1,Y2,...) accepts any number of
%  input variables Y1,Y2,Y3,.... STATINSERTNAN inserts missing values in
%  Y1, Y2,...  and returns them as X1, X2,... respectively.
%
%  This utility is used by some Statistics Toolbox functions to handle
%  missing values.
%
%  See also STATREMOVENAN.


%   Copyright 1993-2008 The MathWorks, Inc.
%   $Revision: 1.1.8.1 $  $Date: 2010/03/16 00:30:15 $

if ~any(wasnan)
     varargout = varargin;
     return;
end

ok = ~wasnan;
len = length(wasnan);
for j=1:nargin-1
    y = varargin{j};
    if (size(y,1)==1) && sum(ok) > 1
        y =  y';
    end
    
    [n,p] = size(y);
    
    if ischar(y)
        x = repmat(' ', [len,p]);
    elseif isa(y, 'nominal')
        x = nominal(NaN([len,p]));
        x = addlevels(x,getlabels(y));
    elseif isa(y, 'ordinal')
        x = ordinal(NaN([len,p]));
        x = addlevels(x,getlabels(y));
    elseif iscell(y)
        x = repmat({''},[len,p]);
    elseif isfloat(y)
            x = nan([len,p],class(y));
    elseif islogical(y)
        error('stats:statinsertnan:InputTypeIncorrect',...
            ['Logical input is not allowed because it can''t '...
            'present missing values. Use CATEGORICAL variable instead.']);
    else
        error('stats:statinsertnan:InputTypeIncorrect',...
            ['Y must be categorical, double, single '...
            ' cell array; or a 2D character array.']);
    end
    
    x(ok,:) = y;
    
    varargout{j} = x;
end
