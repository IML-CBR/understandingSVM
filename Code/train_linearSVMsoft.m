function [ a, u ] = train_linearSVMsoft( labels, data, lambda )
% Funtion for training a SVM that dose consider errors in the cost
% function
    m = size(data,1);
    n = size(data,2)+1;
    X = [data,ones(m,1)];

% Solution via CVX
    cvx_begin
        variables a(n)
        u = max(zeros(m,1),1 - (X*a).*labels);
        minimize (norm(a,2)/2 + lambda*(ones(1,m)*u))
    cvx_end
end

