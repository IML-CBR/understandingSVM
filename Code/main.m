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
model = train_linearSVMhard( labels, data );
model
% 3) and 4)
name = 'linear SVM hard';
plotSVM( data, labels, model, name );

%% QUESTION 2 - Juli
% 1)
Dataset = load('../Data/example_dataset_1');
labels = Dataset.labels;
data = Dataset.data';
% 2)
lambda = 0;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

% 3)
%Created with the toy_datasetCreator function
Dataset = load('../Data/non_separable_dataset_1');
labels = Dataset.labels;
data = Dataset.data';

lambda = 0;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

% 4)
lambda = 0.01;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

lambda = 1;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

lambda = 100;
[model,u] = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

% 5) and 6)
u
% rule for identifying SVs
distances = ([data,ones(size(data,1),1)]*model).*labels
suport_indexes = find(arrayfun(@(x) roundx(x,5,'round'),(distances))==1)
suports = data(suport_indexes,:)



%% QUESTION 3 - Juli
% 1)
Dataset = load('../Data/example_dataset_2');
labels = Dataset.labels;
data = Dataset.data';
% 2) and 3)
lambda = 0.01;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

lambda = 1;
model = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

lambda = 100;
[model,u] = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );
% 4)
lambda = 10;
[model,u] = train_linearSVMsoft( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );
u
% rule for identifying SVs
distances = ([data,ones(size(data,1),1)]*model).*labels
suport_indexes = find(arrayfun(@(x) roundx(x,5,'round'),(distances))==1)
suports = data(suport_indexes,:)

%% QUESTION 4 - Juli
Dataset = load('../Data/example_dataset_2');
labels = Dataset.labels;
data = Dataset.data';
% 2) and 3)
lambda = 0.01;
model = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

lambda = 1;
model = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

lambda = 100;
[model,v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

lambda = 10000;
[model,v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVM( data, labels, model, name );

%% QUESTION 5 - Xavi

%% QUESTION 6 - Xavi