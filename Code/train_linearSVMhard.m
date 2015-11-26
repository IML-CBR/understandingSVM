function [ model, other_values ] = train_linearSVMhard( labels, data, params )
% TODO
    m = size(data,1);
    n = size(data,2);
    A = [data,ones(m,1)];
    cvx_begin
        variable x(n+1)
        minimize( norm( A*x, 2 )/2 )
        subject to
            x'*A*labels >= 1
    cvx_end
end

