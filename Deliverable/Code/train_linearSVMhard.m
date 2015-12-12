function [ w ] = train_linearSVMhard( labels, data )
% Funtion for training a SVM that dose not consider errors in the cost
% function
    m = size(data,1);
    n = size(data,2);
    X = data;
    cvx_begin
        variables w(n) b(1)
        minimize( norm(w,2)/2 )
        subject to
            ((X*w + b).*labels) >= 1;
    cvx_end
    w(end+1) = b;
end

