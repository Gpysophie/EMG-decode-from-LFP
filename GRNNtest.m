%% ������ѷ�������GRNN����
clc;clear;close all;
%% �������ݼ�
load x_train.mat;
load y_train.mat;
load x_test.mat;
load y_test.mat;

%�������Լ�
% x_test=x_train(:,1:720);
% y_test=y_train(:,1:720);


%% ����ģ�����ݼ�
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

%% ������ѷ�������GRNN����
net=newgrnn(desired_input,desired_output,desired_spread);   %ģ��ѵ��
x_test=tramnmx(x_test,xmin,xmax);                           %������������������ʹ����ͬ��һ��
grnn_prediction_result=sim(net,x_test);  
grnn_prediction_result=postmnmx(grnn_prediction_result,ymin,ymax);      %�����һ��
grnn_error=y_test-grnn_prediction_result;
disp(['GRNN������Ԥ������Ϊ',num2str(abs(grnn_error))]);
mse_value=mse(grnn_error);
rmse=sqrt(mse_value);
%����R2
R = corrcoef(grnn_prediction_result,y_test);
R2= R(2)^2;
%save best desired_input desired_output p_test t_test grnn_error mint maxt 
figure(9)
plot(y_test,'-r');
hold on;
plot(grnn_prediction_result,'b');
legend('true','prediction');

