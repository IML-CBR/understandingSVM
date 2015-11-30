function [ w, u ] = train_linearSVMsoft( labels, data, lambda )
% Funtion for training a SVM that dose consider errors in the cost
% function
    m = size(data,1);
    n = size(data,2);
    X = data;
    cvx_begin
        variables w(n) b(1)
        u = max(zeros(m,1),1 - (X*w + b).*labels);
        minimize (norm(w,2)/2 + lambda*(ones(1,m)*u))
    cvx_end
    w(end+1) = b;
end

