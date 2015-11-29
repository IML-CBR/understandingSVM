function [ w ] = train_linearSVMhard( labels, data )
% Funtion for training a SVM that dose not consider errors in the cost
% function
    m = size(data,1);
    n = size(data,2)+1;
    A = [data,ones(m,1)];
    cvx_begin
        variable w(n)
        minimize( norm(w,2)/2 )
        subject to
            ((A*w).*labels) >= 1;
    cvx_end
end

