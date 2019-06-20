clc;
clear;
close all;
%% 路径、配置
rootPath = 'E:\百度云同步盘\任朵朵的自动备份-浙大\任朵朵的文件\matlab\';
% codePath = [rootPath 'test/C02_ROBOT_1D/code/'];
savePath = [rootPath 'LFP2EMGDecode\RNNresult\models\'];
datapath = [rootPath  'LFP2EMGDecode\data_processed'];

addpath(genpath(rootPath));

dirOutput = dir(datapath);
fileNames = {dirOutput.name};
nFiles = length(fileNames);

% 在导入文件时，前两个是空的
nDiscard = 0;
for i = 1 : nFiles
    if length(fileNames{i}) < 3
        nDiscard = nDiscard + 1;
    else
        break;
    end
end
fileNames = fileNames(nDiscard + 1 : end);
nFiles = nFiles - nDiscard;

n = 1; % 1——nFiles

%% 导入数据文件
load(fileNames{n});

%% 对测试集进行最优预测
EMGData=funcNormalization(DataM2.EMGData,0,1);
test_EMGData=EMGData(:,1:720);
test_NeuroFeature=DataM2.NeuroFeature(:,1:720);

load('net_rnn_M1.mat');

%% 测试
prediction_rnn_result=rnn(desired_net, test_NeuroFeature, test_EMGData, [], [], [], [], 1, 0, 0, 0);
prediction_rnn=prediction_rnn_result{1, 1}{1, 4};
figure;
plot(test_EMGData);
hold on;
plot(prediction_rnn);