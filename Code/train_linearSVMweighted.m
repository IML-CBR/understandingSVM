function model = train_linearSVMweighted( labels, data, lambda, w1, w2 )
    % This function trains a weighted SVM
    
    n = size(data,2);
    r = data(labels==-1,:);
    s = data(labels==1,:);
    m = size(s,1);
    k = size(r,1);
    cvx_begin
        variables model(n) b(1) u(m) v(k)
        minimize norm(model,2)+lambda*(ones(1,m)*u*w1+ones(1,k)*v*w2)
        subject to
            (r*model + b) <= -1+v
            (s*model + b) >= 1-u
            u >= 0
            v >= 0
    cvx_end
    model(end+1) = b;
end