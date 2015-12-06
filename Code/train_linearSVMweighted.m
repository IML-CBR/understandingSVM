function model = train_linearSVMweighted( labels, data, lambda )
    % This function trains a weighted SVM
    
    m = size(data,1);
    n = size(data,2);
    r = data(labels==-1,:);
    s = data(labels==1,:);
    cvx_begin
        variables model(n) b(1)
        u = max(zeros(m,1),1 - (data*model + b).*labels);
        minimize norm(model,2)+lambda*(ones(1,m)*u)
        subject to
            (r*model + b) <= -1
            (s*model + b) >= 1-u
            u >= 0
    cvx_end
    model(end+1) = b;
end