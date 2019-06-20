clc;clear;close all;
%% 载入数据集
% M2
load 'NeuroFeature_M2.mat';
load 'MoveData_M2.mat'
load 'EMGData_M2.mat'
% M3
% load 'NeuroFeature_M3.mat';
% load 'MoveData_M3.mat'
% load 'EMGData_M3.mat'

%% 运动数据归一化
EMGData=funcNormalization(EMGData(1,:),0,1);  %归一到0-100之间


%% 生成python数据集
% NeuroFeature=NeuroFeature';
% MoveData=MoveData';
% LFPDataset=[NeuroFeature,MoveData];
% 
% train_data=LFPDataset(1:9000,:);
% test_data=LFPDataset(9001:end,1:2052);
% 
% [trainedModel, validationRMSE] = trainRegressionModel(train_data);
% yfit=trainedModel.predictFcn(test_data);
% plot(yfit);
% hold on
% plot(LFPDataset(9001:end,33));
% csvwrite('LFPDataset.csv',LFPDataset);

%% PCA 降维
% [pc,score,latent,tsquare] = princomp(NeuroFeature');
% variance_ratio=cumsum(latent)./sum(latent);
% Kcomp=0;
% for i=1:size(variance_ratio,1)
%     if variance_ratio(i)>=0.98
%         Kcomp=i;
%         break;
%     end
% end
% NeuroFeatureReduceDim=score(:,1:Kcomp)';

%% 分为训练和预测两类
x_train=NeuroFeature(:,720:end);
y_train=EMGData(:,720:end);
x_test=NeuroFeature(:,1:720);
y_test=EMGData(:,1:720);
csvwrite('x_train.csv',x_train);
csvwrite('y_train.csv',y_train);
csvwrite('x_test.csv',x_test);
csvwrite('y_test.csv',y_test);


%% 交叉验证
nfold=10;
[desired_input,desired_output,desired_spread,desired_xmin,desired_xmax,desired_ymin,desired_ymax,all_mse,all_cc]...
    =funcGRNN(nfold,x_train,y_train,x_test,y_test);












