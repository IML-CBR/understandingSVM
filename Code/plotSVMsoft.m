function plotSVMsoft( data, labels, model, name )
% Function that plots the SVM model generated for the training data
    linewidth = 0.5;
    range_min = min(data(:,1));
    range_max = max(data(:,1));
    offset = abs(range_max-range_min)/10;
    range = linspace(range_min-offset,range_max+offset,100);
    w_lim = (model(3)+model(1)*range)/(-model(2));
    marg1 = (model(3)+model(1)*range + 1)/(-model(2));
    marg2 = (model(3)+model(1)*range - 1)/(-model(2));
    
    positive = data(find(labels > 0),:);
    negative = data(find(labels < 0),:);
    graph = plot(positive(:,1),positive(:,2), '+', negative(:,1), negative(:,2), 'o');
    set(graph(1),'LineWidth',linewidth);
    set(graph(2),'LineWidth',linewidth);
    set(graph(2),'MarkerFaceColor',[0 0.5 0]);
    hold on;
    plot(range,w_lim, '-b', range,marg1, '--r', range,marg2, '--r');
    hold on;
%     aux4 = [data,ones(size(data,1),1)];
%     aux3 = aux4*model;
%     aux2 = sum(aux3);
%     aux1 = abs(sum([data,ones(size(data,1),1)]*model))/((sum(model(1:2).^2)).^0.5);
% 
%     aux = arrayfun(@(x) roundx(x,5,'round'),(abs(sum([data,ones(size(data,1),1)]*model))/((sum(model(1:2).^2)).^0.5)));
%     suports = data(find(arrayfun(@(x) roundx(x,5,'round'),...
%         (aux1))<=1),:);
%    aux = (((([data,ones(size(data,1),1)]*model))));
%    aux1=sum(model(1:2).^2).^0.5;
 aux = (abs([data,ones(size(data,1),1)]*model)/...
        ((sum(model(1:2).^2)).^0.5));
    suports = data(find(arrayfun(@(x) roundx(x,5,'round'),...
        ((abs([data,ones(size(data,1),1)]*model)/...
        ((model(1).^2+model(2).^2).^0.5))))<=1),:);
    scatter(suports(:,1),suports(:,2),200,'g','o','LineWidth',1.5);
    axis tight
    title(name)
    xlabel('x1')
    ylabel('x2')
    hold off


%     minrange = min(data,[],1);
%     maxrange = max(data,[],1);
%     range = [minrange(1),maxrange(1),minrange(2),maxrange(2)];
%     fun = @(x) ((model(1)+model(2)*x)/(-model(3)));
%     funMar1 = @(x) ((1+model(1)+model(2)*x)/(-model(3)));
%     funMar2 = @(x) ((-1+model(1)+model(2)*x)/(-model(3)));
%     positive = data(find(labels > 0),:);
%     negative = data(find(labels < 0),:);
%     figure;
% 
%     scatter(positive(:,1),positive(:,2),[],'b','+');
%     hold on
%     scatter(negative(:,1),negative(:,2),[],'r','o');
%     hold on
%     fplot(fun,range);
%     hold on
%     fplot(funMar1,range);
%     hold on
%     fplot(funMar2,range);
% 
%     suports = data(find(dist<=1),:);
%     hold on
%     scatter(suports(:,1),suports(:,2),200,'g','o','LineWidth',1.5);
% 
%     title(name)
%     xlabel('x1')
%     ylabel('x2')
%     hold off
end

