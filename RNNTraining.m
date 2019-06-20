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

n = 2; % 1——nFiles
savePath = [savePath 'M3' '\'];
% rng(1234);

%% 参数设置

layerTypes = {'linear', 'recttanh', 'linear' };
layerSizes = [28 64 64 1];      %输入维度，层数，层数，输出维度
objectiveFunction = 'sum-of-squares';

tau = 0.1;  % 100ms             dt/tau越小，前面取得值越多，越平滑
dt = 0.04;  % 20ms，1个bin的大小 0.04
numconn = 8;

% damping
mu = 2e-3;
lambda = 3e-4;

l2weightcost = 1e-3;    % weight cost1e-4
g_by_layer = ones(length(layerSizes), 1);
g_by_layer(2) = 0.9;
cm_fac_by_layer = ones(3, 1);

% 优化器参数
objfuntol = 3e-7;
mincgiter = 10;                 % 大于10
maxcgiter = 99;
cgepsilon = 8e-8;
max_hf_iters = 200;             % 总迭代次数
max_consec_failures = 50;       % 当前迭代效果比上一次差，视为失败，连续失败50次退出
max_failures = 100;             % 一共失败100次退出
cgbt_objfun = 'train';

% 其他
kfold = 10; % 10折交叉验证

% 并行计算相关
bParCalculateObj = false;
bParCalculateGrad = false;
bParCalculateCG = false;
% 作图函数
OptionalPlotFun = @funcCCPlot;
simparams.init = 1;

saves.path = savePath;
saves.interval = 2;

%% 训练网络
% 需要改正交参数‘BOrth’
net = rnn_init(layerSizes, layerTypes, g_by_layer, objectiveFunction, ...
        'cmFactors', cm_fac_by_layer, 'numconn',  numconn, ... 
        'tau', tau, 'dt', dt, 'mu', mu, 'bOrth', false);

% 导入数据
load(fileNames{n});
EMGData=funcNormalization(DataM3.EMGData,0,1);  %归一到0-100之间
EMGData=EMGData(:,720:end);
NeuroFeature=DataM3.NeuroFeature(:,720:end);
% [chns, bins] = size(allSpk);
% allSpk = allSpk + randn(chns, bins);
% save([fileNames{n}(1 : 17) '(noise).mat'], 'allSpk', 'allPos');

% 交叉验证
cc = 0;
CCs = zeros(1, kfold);
mse_value = 0;
MSEs = zeros(1, kfold);

mse_min=10e20;
desired_net=[];


for j = 1 : kfold
% for j = 7
    close all
    % 保存模型名
    if j < kfold
        saves.name = [fileNames{n} '_model_0' num2str(j)];
    elseif j == kfold
        saves.name = [fileNames{n} '_model_' num2str(j)];
    end
    
    l = length(EMGData);
    ltest = floor(l / kfold);
    v_neuro = NeuroFeature(:, 1 + (j - 1) * ltest : j * ltest);
    v_emg = EMGData(1, 1 + (j - 1) * ltest : j * ltest);
    t_neuro = NeuroFeature(:, [1 : (j - 1) * ltest j * ltest + 1 : end]);
    t_emg = EMGData(1, [1 : (j - 1) * ltest j* ltest + 1 : end]);
    
    minibatchSize = length(t_neuro);
    
    [opttheta, objfun_train, objfun_test, stats] = hfopt3(net, ...
        t_neuro, t_emg, v_neuro, v_emg, max_hf_iters, saves, true, ...
        'bParCalculateObj', bParCalculateObj, 'bParCalculateGrad', bParCalculateGrad, 'bParCalculateCG', bParCalculateCG, ...
        'batchSize', minibatchSize, 'objectTol', objfuntol, 'cgMaxFailures', max_failures, ...
        'hfmaxconsecutivefailures', max_consec_failures, ...
        'cgMinIter', mincgiter, 'cgMaxIter', maxcgiter, 'cgTolerance', cgepsilon, ...
        'lambda', lambda, 'weightCost', l2weightcost, 'paramsToSave', simparams, ...
        'optPlotFun', OptionalPlotFun);
    
    testnet = net;
    testnet.theta = opttheta;
    
    forwardPass = rnn(testnet, v_neuro, v_emg, [], [], [], [], 1, 0, 0, 0);
    cc = corrcoef(v_emg, forwardPass{1, 1}{1, 4});
    mse_value = mse(v_emg, forwardPass{1, 1}{1, 4});
    
    if mse_value<mse_min
        mse_min=mse_value;
        desired_net=testnet;
    end
    
    CCs(j) = cc(1, 2);
    MSEs(j) = mse_value;
end

mean_cc = mean(CCs);
mean_mse = mean(MSEs);

