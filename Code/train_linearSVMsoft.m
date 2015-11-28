function [ a ] = train_linearSVMsoft( labels, data, lambda )
% Funtion for training a SVM that dose consider errors in the cost
% function
    m = size(data,1);
    n = size(data,2)+1;
    X = [data,ones(m,1)];
%     cvx_begin 
%         variable w(n)
%         u = zeros(1,m)
%         for i=1:m
% %             u(i) = max(0,1-(A(i,:)*w)*labels(i));%(u(i)*labels(i)).^0.5
%         end
%         minimize( norm( w, 2 )/2 + lambda*(sum(max(0,X*w*labels))) )
%     cvx_end
    
    
%     g = 0.1;            % gamma

% Solution via CVX
    cvx_begin
        variables a(n)
        minimize (norm(a,2)/2 + lambda*(ones(1,m)*max(zeros(m,1),1 - (X*a).*labels)))
%         X'*a - b >= 1 - u;
%         u = max(zeros(m,1),1 - (X*a).*labels)
%         Y'*a - b <= -(1 - v);
%         u >= 0;
    cvx_end
%     for i=1:m
%         u(i) = roundx(u(i),5,'round');
%     end
end

