%% 采用最佳方法建立GRNN网络
clc;clear;close all;
%% 载入数据集
load x_train.mat;
load y_train.mat;
load x_test.mat;
load y_test.mat;

%更换测试集
% x_test=x_train(:,1:720);
% y_test=y_train(:,1:720);


%% 载入模型数据集
load desired_input.mat;
load desired_output.mat;
load desired_spread.mat;
load desired_xmin.mat;
load desired_xmax.mat;
load desired_ymin.mat;
load desired_ymax.mat;

xmin=desired_xmin;
xmax=desired_xmax;
ymin=desired_ymin;
ymax=desired_ymax;

%% 采用最佳方法建立GRNN网络
net=newgrnn(desired_input,desired_output,desired_spread);   %模型训练
x_test=tramnmx(x_test,xmin,xmax);                           %网络数据与样本数据使用相同归一化
grnn_prediction_result=sim(net,x_test);  
grnn_prediction_result=postmnmx(grnn_prediction_result,ymin,ymax);      %反向归一化
grnn_error=y_test-grnn_prediction_result;
disp(['GRNN神经网络预测的误差为',num2str(abs(grnn_error))]);
mse_value=mse(grnn_error);
rmse=sqrt(mse_value);
%计算R2
R = corrcoef(grnn_prediction_result,y_test);
R2= R(2)^2;
%save best desired_input desired_output p_test t_test grnn_error mint maxt 
figure(9)
plot(y_test,'-r');
hold on;
plot(grnn_prediction_result,'b');
legend('true','prediction');

