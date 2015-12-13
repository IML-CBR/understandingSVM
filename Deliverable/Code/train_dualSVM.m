function [  model, v ] = train_dualSVM( labels, data, lambda )
% Function for training a SVM that does consider errors in the cost
% function, with the dual algorithm solution
    m = size(data,1);
    n = size(data,2);
    X = data';
    Y = labels;
    K = X'*X;
    tolerance = 1e-4;
    cvx_begin
        variables v(m)
        maximise(v'*ones(m,1)-(1/2)*ones(1,m)*(v'*diag(Y)*K*diag(Y)*v)*ones(m,1));
        subject to
            v'*Y == 0;
            lambda >= v;
            v >= 0;
    cvx_end
    
% Fix v values before recovering primal representation: Y = w^T * X + b
%     finish = 0;
%     decimals = 0;
%     while (decimals < 6) && ~finish %&& (lambda/10^decimals > mean(v)/100)
%         finish = (size(Y,1) - (sum(v <= lambda/10^decimals)+sum((arrayfun(@(x) roundx(x,decimals,'round'),v) == lambda))) == 3);
%         if ~finish decimals = decimals+1; end
%     end
%     
%     v(find(v <= lambda/10^(decimals))) = 0;
%     v(find((arrayfun(@(x) roundx(x,decimals,'round'),v)) == lambda)) = lambda;
%
% Recover w from the dual
%     w = (v'.*Y'*X')';
   
    
    model.margin = find(v > tolerance * lambda & v < (1 - tolerance) * lambda) ;
    model.svs = find(v > tolerance * lambda) ;
    model.v = v ;
    model.vy = diag(Y) * v ;

if ~ isempty(model.margin)
  % This works almost all times
  model.b = 1-mean(Y(model.margin)'.*(Y(model.margin)' - model.vy' * K(:,model.margin))) ;

else
  % Special cases to deal with the case in which have a very small C
  % and there are no support vectors on the margin

  r = 1 - model.vy' * K * diag(Y) ;
  act = ismember(1:n, model.svs) ;
  pos = Y > 0 ;

  maxb = min([+r(pos & act),  -r(~pos & ~act)]) ;
  minb = max([-r(~pos & act), +r(pos & ~act)]) ;
  if mean(Y(act)) <= -tolerance
    model.b = maxb ;
  elseif mean(Y(act)) > tolerance
    model.b = minb;
  else
    % Any b in the interval is equivalent
    model.b = mean([minb maxb]) ;
  end
    
end

