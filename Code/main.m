%% Reset all
clear all;
close all;
clc;
%% Move to working directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
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
fun = @(x) ((model(3)+model(1)*x)/(-model(2)));
funMar1 = @(x) ((1+model(3)+model(1)*x)/(-model(2)));
funMar2 = @(x) ((-1+model(3)+model(1)*x)/(-model(2)));
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
% 1)
Dataset = load('../Data/example_dataset_1');
labels = Dataset.labels;
data = Dataset.data';
% 2)
lambda = 0;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('SVM soft with lambda ',num2str(lambda));
plotSVMsoft( data, labels, model, name );

% 3)
%Created with the toy_datasetCreator function
Dataset = load('../Data/non_separable_dataset_3');
labels = Dataset.labels;
data = Dataset.data';

lambda = 0;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('SVM soft with lambda ',num2str(lambda));
plotSVMsoft( data, labels, model, name );

% 4)
lambda = 0.01;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('SVM soft with lambda ',num2str(lambda));
plotSVMsoft( data, labels, model, name );

lambda = 1;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('SVM soft with lambda ',num2str(lambda));
plotSVMsoft( data, labels, model, name );

lambda = 100;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('SVM soft with lambda ',num2str(lambda));
plotSVMsoft( data, labels, model, name );

%% QUESTION 3 - Juli

%% QUESTION 4 - Optional

%% QUESTION 5 - Xavi

%% QUESTION 6 - Xavi