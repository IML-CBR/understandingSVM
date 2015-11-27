%% Reset all
clear all;
close all;
%% Move to working directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
%% Load Data


%% QUESTION 1 - Juli
% 1)
Dataset = load('../Data/example_dataset_1');
labels = Dataset.labels;
data = Dataset.data';
% 2)
 [model,dist] = train_linearSVMhard( labels, data );
% 3)
minrange = min(data,[],1);
maxrange = max(data,[],1);
range = [minrange(1),maxrange(1),minrange(2),maxrange(2)];
fun = @(x) ((model(1)+model(2)*x)/(-model(3)));
funMar1 = @(x) ((1+model(1)+model(2)*x)/(-model(3)));
funMar2 = @(x) ((-1+model(1)+model(2)*x)/(-model(3)));
positive = data(find(labels > 0),:);
negative = data(find(labels < 0),:);
figure;

scatter(positive(:,1),positive(:,2),[],'b','+');
hold on
scatter(negative(:,1),negative(:,2),[],'r','o');
hold on
fplot(fun,range);
hold on
fplot(funMar1,range);
hold on
fplot(funMar2,range);

% 4)
suports = data(find(dist==1),:);
hold on
scatter(suports(:,1),suports(:,2),200,'g','o','LineWidth',1.5);

title('Analytical solution')
xlabel('x1')
ylabel('x2')
hold off

%% QUESTION 2 - Juli

%% QUESTION 3 - Juli

%% QUESTION 4 - Optional

%% QUESTION 5 - Xavi

%% QUESTION 6 - Xavi