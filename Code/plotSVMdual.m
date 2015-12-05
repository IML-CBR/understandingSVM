function plotSVMdual( data, labels, model, name, d )
% Function that plots the SVM model generated for the training data
    figure;
    linewidth = 0.5;
    range_min = min(data(:,1));
    range_max = max(data(:,1));
    offset = abs(range_max-range_min)/10;
    range = linspace(range_min-offset,range_max+offset,100);
    w_lim = (model(1)*range)/(-model(2));
    marg1 = (model(1)*range + mean(d))/(-model(2));
    marg2 = (model(1)*range - mean(d))/(-model(2));
    
    positive = data(find(labels > 0),:);
    negative = data(find(labels < 0),:);
    graph = plot(positive(:,1),positive(:,2), '+', negative(:,1), negative(:,2), 'o');
    set(graph(1),'LineWidth',linewidth);
    set(graph(1),'MarkerFaceColor',[0 0 0.5]);
    set(graph(2),'LineWidth',linewidth);
    set(graph(2),'Color',[0.5 0 0]);
    hold on;
    
    % Distance is calculated as r = (w'*X_i + b)
    distances = (data*model).*labels;
    supIndex = (unique([find(distances==d(1)),find(distances==d(2)),find(distances==d(3))]'))';
    suports = data(supIndex,:);
%     while length(suports)>3
%         decimals = decimals + 1;
%         suports = data(find(arrayfun(@(x) roundx(x,decimals,'round'),(distances))==d),:);
%     end
    errIndex = intersect(intersect(find(distances < d(1)),find(distances < d(2))),find(distances < d(3)));
    errors = data(errIndex,:);
    
    
    scatter(errors(:,1),errors(:,2),200,'y','o','LineWidth',1.5);
    hold on
    scatter(suports(:,1),suports(:,2),200,'g','o','LineWidth',1.5);
    
    
    hold on;
    plot(range,w_lim, '-b', range,marg1, '--r', range,marg2, '--r');
    axis tight
    title(name)
    
    xlabel('x1')
    ylabel('x2')
    hold off
    
%Real distance is not used
%     suports = data(find(arrayfun(@(x) roundx(x,5,'round'),...
%         ((abs([data,ones(size(data,1),1)]*model)/...
%         ((model(1).^2+model(2).^2).^0.5))))<=1),:);

end

