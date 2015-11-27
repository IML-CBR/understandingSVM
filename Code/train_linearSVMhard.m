function [ w , d ] = train_linearSVMhard( labels, data )
% TODO
    m = size(data,1);
    n = size(data,2)+1;
    A = [ones(m,1),data];
%     d = zeros(1,20);
    cvx_begin 
        variable w(n)
        minimize( norm( w, 2 )/2 )
        d = A*w
        for i=1:m
            d(i) = (d(i)*labels(i)).^0.5
        end
        subject to
            d >= 1
    cvx_end
    for i=1:m
        d(i) = roundx(d(i),5,'round');
    end
end

