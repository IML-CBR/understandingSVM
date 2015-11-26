%% Reset all
clear all;
close all;
%% Move to working directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
%% Load Data


%% QUESTION 1 - Juli
% 1)
DataSet = load('../Data/example_dataset_1');
labels = DataSet.labels;
data = Dataset.data;
% 2)
[ model, other_values ] = train_linearSVMhard( labels, data, params );
% 3)
% 4)

%% QUESTION 2 - Juli

%% QUESTION 3 - Juli

%% QUESTION 4 - Optional

%% QUESTION 5 - Xavi

%% QUESTION 6 - Xavi