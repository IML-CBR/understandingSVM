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
[model, v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVMdual( data, labels, model, name );

lambda = 1;
[model, v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVMdual( data, labels, model, name );

lambda = 100;
[model, v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVMdual( data, labels, model, name );

% 4)
lambda = 10;
[model, v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVMdual( data, labels, model, name );

% 5) & 6)
v
v(find(v > 0))
% SVs:
data(find(v > 0),:)

%% QUESTION 5 - Xavi
dataset_5 = load('../Data/example_dataset_3');
data_5 = dataset_5.data';
labels_5 = dataset_5.labels;

% Num examples each class
label_pos = sum(labels_5==1)
label_neg = sum(labels_5==-1)

decimals = 5;
lambdas = [0.1 0 1 10 100];
best_error = Inf;
for i = 1:length(lambdas)
    lambda = lambdas(i);
    model_5 = train_linearSVMsoft( labels_5, data_5, lambda );
    distances = ([data_5,ones(size(data_5,1),1)]*model_5).*labels_5;
    errors = data_5(find(arrayfun(@(x) roundx(x,decimals,'round'),(distances))<1),:);
    if size(errors,1) < best_error
        best_error = size(errors,1);
        best_model = model_5;
        best_lambda = lambda;
    end
end

distances = ([data_5,ones(size(data_5,1),1)]*best_model).*labels_5;
error_labels = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<1);
plotSVM( data_5, labels_5, best_model, name );
    
%% QUESTION 6 - Xavi

model_5 = train_linearSVMweighted( labels_5, data_5, lambda );
plotSVM( data_5, labels_5, model_5, name );
