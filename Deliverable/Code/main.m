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
% 1)
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
tolerance = 1e-4; % we need a threshold in order to ignore residual values 
                    % from the minimitzation process that should not be 
                    % considered greater than zero, otherwise, we will 
                    % probably retrive all vi values

% v values for lambda = 10
v

% indexes of the vi of the SVs:
indexSVs = find(v > tolerance * lambda) % we use lambda as part of the threshold 
    % in orther to relate tolerance to the v values magnitude, because all 
    % v values will allways be within the range 0 <= v <= lambda
    
% vi of the SVs:
vSVs = v(find(v > tolerance * lambda))

% SVs:
SVs = data(find(v > tolerance * lambda),:)


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
        best_error_block_5 = size(errors_svm,1);
        best_model_block_5 = model_5;
        best_lambda_block_5 = lambda;
    end
end


distances = ([data_5,ones(size(data_5,1),1)]*best_model_block_5).*labels_5;
error_svm_labels = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<1);
error_labels = labels_5(arrayfun(@(x) roundx(x,decimals,'round'),distances)<0);
plotSVM( data_5, labels_5, best_model_block_5, 'Soft SVM with unbalanced data' );


%% QUESTION 6
dataset_6 = load('../Data/example_dataset_3');
data_6 = dataset_6.data';
labels_6 = dataset_6.labels;

% Num examples each class
num_pos = sum(labels_6==1)
num_neg = sum(labels_6==-1)

w1 = 1-num_pos/(num_pos+num_neg)
w2 = 1-w1

decimals_6 = 5;
lambdas_6 = [1 10 100 1000 10000];
best_error_6 = Inf;
errors_svm_6 = zeros(1,length(lambdas_6));
errors_mis_6 = zeros(1,length(lambdas_6));
for i = 1:length(lambdas_6)
    lambda = lambdas_6(i);
    model_6 = train_linearSVMweighted( labels_6, data_6, lambda, w1,w2 );
    distances = ([data_6,ones(size(data_6,1),1)]*model_6).*labels_6;
    error_svm = labels_6(arrayfun(@(x) roundx(x,decimals_6,'round'),distances)<1);
    error_mis = labels_6(arrayfun(@(x) roundx(x,decimals_6,'round'),distances)<0);
    errors_svm_6(i) = size(error_svm,1);
    errors_mis_6(i) = size(error_mis,1);
    if size(error_svm,1) < best_error_6
        best_error_6 = size(error_svm,1);
        best_model_6 = model_6;
        best_lambda_6 = lambda;
    end
end
distances = ([data_6,ones(size(data_6,1),1)]*best_model_6).*labels_6;
error_svm_labels = labels_6(arrayfun(@(x) roundx(x,decimals_6,'round'),distances)<1);
error_labels = labels_6(arrayfun(@(x) roundx(x,decimals_6,'round'),distances)<0);

plotSVM( data_6, labels_6, best_model_6, 'Weighted SVM with unbalanced data' );


%% Weighted error
% Block 5
distances_5 = ([data_5,ones(size(data_5,1),1)]*best_model_block_5).*labels_5;

list_svm_errors_5 = labels_5(distances_5<1);
error_svm_5 = size(list_svm_errors_5,1)/size(data_5,1);

list_misclassified_5 = labels_5(distances_5<0);
error_misclassified_5 = size(list_misclassified_5,1)/size(data_5,1);

weighted_error_5 = sum((list_svm_errors_5==1)*w1+(list_svm_errors_5==-1)*w2);
weighted_error_5 = weighted_error_5 / sum((labels_5==1)*w1+(labels_5==-1)*w2);

error_svm_5
error_misclassified_5
weighted_error_5

% Block 6
distances_6 = ([data_6,ones(size(data_6,1),1)]*best_model_6).*labels_6;
list_svm_errors_6 = labels_6(distances_6 < 1);
error_svm_6 = size(list_svm_errors_6,1)/size(data_6,1);

list_misclassified_6 = labels_6(distances_6<0);
error_misclassified_6 = size(list_misclassified_6,1)/size(data_6,1);

weighted_error_6 = sum((list_svm_errors_6==1)*w1+(list_svm_errors_6==-1)*w2);
weighted_error_6 = weighted_error_6 / sum((labels_6==1)*w1+(labels_6==-1)*w2);

error_svm_6
error_misclassified_6
weighted_error_6