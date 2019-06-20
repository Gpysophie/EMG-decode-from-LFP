clc;
clear;
close all;
%% ·��������
rootPath = 'E:\�ٶ���ͬ����\�ζ����Զ�����-���\�ζ����ļ�\matlab\';
% codePath = [rootPath 'test/C02_ROBOT_1D/code/'];
savePath = [rootPath 'LFP2EMGDecode\RNNresult\models\'];
datapath = [rootPath  'LFP2EMGDecode\data_processed'];

addpath(genpath(rootPath));

dirOutput = dir(datapath);
fileNames = {dirOutput.name};
nFiles = length(fileNames);

% �ڵ����ļ�ʱ��ǰ�����ǿյ�
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

n = 2; % 1����nFiles
savePath = [savePath 'M3' '\'];
% rng(1234);

%% ��������

layerTypes = {'linear', 'recttanh', 'linear' };
layerSizes = [28 64 64 1];      %����ά�ȣ����������������ά��
objectiveFunction = 'sum-of-squares';

tau = 0.1;  % 100ms             dt/tauԽС��ǰ��ȡ��ֵԽ�࣬Խƽ��
dt = 0.04;  % 20ms��1��bin�Ĵ�С 0.04
numconn = 8;

% damping
mu = 2e-3;
lambda = 3e-4;

l2weightcost = 1e-3;    % weight cost1e-4
g_by_layer = ones(length(layerSizes), 1);
g_by_layer(2) = 0.9;
cm_fac_by_layer = ones(3, 1);

% �Ż�������
objfuntol = 3e-7;
mincgiter = 10;                 % ����10
maxcgiter = 99;
cgepsilon = 8e-8;
max_hf_iters = 200;             % �ܵ�������
max_consec_failures = 50;       % ��ǰ����Ч������һ�β��Ϊʧ�ܣ�����ʧ��50���˳�
max_failures = 100;             % һ��ʧ��100���˳�
cgbt_objfun = 'train';

% ����
kfold = 10; % 10�۽�����֤

% ���м������
bParCalculateObj = false;
bParCalculateGrad = false;
bParCalculateCG = false;
% ��ͼ����
OptionalPlotFun = @funcCCPlot;
simparams.init = 1;

saves.path = savePath;
saves.interval = 2;

%% ѵ������
% ��Ҫ������������BOrth��
net = rnn_init(layerSizes, layerTypes, g_by_layer, objectiveFunction, ...
        'cmFactors', cm_fac_by_layer, 'numconn',  numconn, ... 
        'tau', tau, 'dt', dt, 'mu', mu, 'bOrth', false);

% ��������
load(fileNames{n});
EMGData=funcNormalization(DataM3.EMGData,0,1);  %��һ��0-100֮��
EMGData=EMGData(:,720:end);
NeuroFeature=DataM3.NeuroFeature(:,720:end);
% [chns, bins] = size(allSpk);
% allSpk = allSpk + randn(chns, bins);
% save([fileNames{n}(1 : 17) '(noise).mat'], 'allSpk', 'allPos');

% ������֤
cc = 0;
CCs = zeros(1, kfold);
mse_value = 0;
MSEs = zeros(1, kfold);

mse_min=10e20;
desired_net=[];


for j = 1 : kfold
% for j = 7
    close all
    % ����ģ����
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

