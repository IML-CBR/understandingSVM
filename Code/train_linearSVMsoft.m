function [ w , d ] = train_linearSVMsoft( labels, data, lambda )
% Funtion for training a SVM that dose consider errors in the cost
% function
    m = size(data,1);
    n = size(data,2)+1;
    A = [ones(m,1),data];
    cvx_begin 
        variable w(n)
        d = A*w
        for i=1:m
            d(i) = (d(i)*labels(i)).^0.5
        end
        minimize( norm( w, 2 )/2 + lambda*(sum(max(0, 1 - d))) )
    cvx_end
    for i=1:m
        d(i) = roundx(d(i),5,'round');
    end
end

