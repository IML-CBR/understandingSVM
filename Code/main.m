%% Reset all
clear all;
close all;
clc;

%% Move to working directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));


%% QUESTION 1
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


%% QUESTION 2
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


%% QUESTION 3
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


%% QUESTION 4
Dataset = load('../Data/example_dataset_2');
labels = Dataset.labels;
data = Dataset.data';
% 2) and 3)
lambda = 0.01;
[model, v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVMdual( data, labels, model, name, v, lambda );

lambda = 1;
[model, v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVMdual( data, labels, model, name, v, lambda );

lambda = 100;
[model, v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVMdual( data, labels, model, name, v, lambda );

% 4)
lambda = 10;
[model, v] = train_dualSVM( labels, data, lambda );
name = strcat('linear SVM soft with lambda ',num2str(lambda));
plotSVMdual( data, labels, model, name, v, lambda );

% 5) & 6)
v
v(find(v > 0))
% SVs:
data(find(v > 0),:)


%% QUESTION 5
dataset_5 = load('../Data/example_dataset_3');
data_5 = dataset_5.data';
labels_5 = dataset_5.labels;

% Num examples each class
num_pos = sum(labels_5==1)
num_neg = sum(labels_5==-1)

decimals = 5;
lambdas = [1 10 100 1000 10000];
errors_t = zeros(1,length(lambdas));
best_error = Inf;
for i = 1:length(lambdas)
    lambda = lambdas(i);
    model_5 = train_linearSVMsoft( labels_5, data_5, lambda );
    distances = ([data_5,ones(size(data_5,1),1)]*model_5).*labels_5;
    
    error_svm = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<1);
    error_mis = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<0);
    errors_svm(i) = size(error_svm,1);
    errors_mis(i) = size(error_mis,1);
    if size(errors_svm,1) < best_error
        best_error = size(errors_svm,1);
        best_model = model_5;
        best_lambda = lambda;
    end
end


distances = ([data_5,ones(size(data_5,1),1)]*best_model).*labels_5;
error_svm_labels = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<1);
error_labels = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<0);
plotSVM( data_5, labels_5, best_model, 'Soft SVM with unbalanced data' );


%% QUESTION 6
dataset_5 = load('../Data/example_dataset_3');
data_5 = dataset_5.data';
labels_5 = dataset_5.labels;

% Num examples each class
num_pos = sum(labels_5==1)
num_neg = sum(labels_5==-1)

w1 = 1-num_pos/(num_pos+num_neg)
w2 = 1-w1

decimals = 5;
lambdas = [1 10 100 1000 10000];
best_error = Inf;
errors_svm = zeros(1,length(lambdas));
errors_mis = zeros(1,length(lambdas));
for i = 1:length(lambdas)
    lambda = lambdas(i);
    model_5_w = train_linearSVMweighted( labels_5, data_5, lambda, w1,w2 );
    distances = ([data_5,ones(size(data_5,1),1)]*model_5_w).*labels_5;
    error_svm = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<1);
    error_mis = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<0);
    errors_svm(i) = size(error_svm,1);
    errors_mis(i) = size(error_mis,1);
    if size(error_svm_labels,1) < best_error
        best_error = size(error_svm_labels,1);
        best_model = model_5_w;
        best_lambda = lambda;
    end
end
distances = ([data_5,ones(size(data_5,1),1)]*best_model).*labels_5;
error_svm_labels = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<1);
error_labels = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<0);

plotSVM( data_5, labels_5, best_model, 'Weighted SVM with unbalanced data' );
