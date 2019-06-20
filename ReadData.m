clc;clear;close all;
%% ���ö�ȡ��Ϣ
settings.fs = 2000;             % ������2000
settings.emgCh = 1:2;           % �����ź�ͨ��
settings.LFPCh = 3:6;           % ���ź�ͨ��
settings.Move = 7;              % �����ź�ͨ��
%% ���������ݺ��˶�����
rootPath='E:\�ٶ���ͬ����\�ζ����Զ�����-���\�ζ����ļ�\matlab\LFP2EMGDecode';
addpath(genpath(rootPath));
load datafile20190510M3;
DataFile = datafile20190510M3;   %�����ļ�
%temp=size(datafile20190416M2.Data,2);
%emg����
EMGDataRead = DataFile.Data(settings.emgCh,:);
EMGDataRead = double(EMGDataRead);
%������
NeuroDataRead = DataFile.Data(settings.LFPCh,:);%��ȡ1-4���� 
NeuroDataRead = double(NeuroDataRead);%ת��Ϊ������
%�˶�����
MoveDataRead=DataFile.Data(settings.Move,:);%��ȡ�˶�����
MoveDataRead = double(MoveDataRead);%ת��Ϊ������

%% ������
% for i=settings.SelCh
%     NeuroDataRaw(i,:)=resample(NeuroDataRead(i,:),1000,2000);
% end
% MoveDataRaw=resample(MoveDataRead,1000,2000);
% 
% settings.fs=1000;

%% ȥ����Ư��-����ʽ���ȥ����Ư��
% polyorder=6;
% MoveDataDetrend=funcPolyfit(MoveData_Filter,polyorder);
% % ȥ����ֵ
% for i = 1:length(MoveDataDetrend)
%     MoveDataDetrend(i) = (MoveDataDetrend(i)>=0)*MoveDataDetrend(i);
% end

MoveDataDetrend=detrend(MoveDataRead,'constant');
% ��������
for i = 1:length(MoveDataDetrend)
    MoveDataDetrend(i) = (MoveDataDetrend(i)>=0)*MoveDataDetrend(i);
end
figure(11);
plot(MoveDataDetrend,'r')
hold on;
plot(MoveDataRead,'b')

%% �˶����ݵ�ͨ�˲�
[b,a]=butter(2,2*20/settings.fs,'low');
MoveData_Filter=filter(b,a,MoveDataDetrend);
MoveData_Filter=smooth(MoveData_Filter); %ƽ���˲�
MoveData_Filter=MoveData_Filter';
figure(1);
plot(MoveDataDetrend,'r');
hold on;
plot(MoveData_Filter,'b')
% ȥ����ֵ
baseline_lever=mean(MoveData_Filter(1,size(MoveData_Filter,2)-1000:end));
for i = 1:length(MoveData_Filter)
    if(MoveData_Filter(i)<baseline_lever)
        MoveData_Filter(i) = baseline_lever;
    end
end
figure(2);
plot(MoveDataDetrend,'r');
hold on;
plot(MoveData_Filter,'b')



%% EMG�ź�Ԥ����
% calculate emg
ch1=EMGDataRead(1,:);
ch2=EMGDataRead(2,:);
EMGDataRaw=ch1-ch2;        %double electrodes method
% filt the emg data
% filt 50Hz
fltEMG1=funcNotchFilter(EMGDataRaw,50,settings.fs);
% filt 100Hz
fltEMG2=funcNotchFilter(fltEMG1,100,settings.fs);
% filt 150Hz
fltEMG3=funcNotchFilter(fltEMG2,150,settings.fs);
% filt 200Hz
fltEMG4=funcNotchFilter(fltEMG3,200,settings.fs);
% filt 250Hz
fltEMG5=funcNotchFilter(fltEMG4,300,settings.fs);
% filt 300Hz
fltEMG6=funcNotchFilter(fltEMG5,400,settings.fs);
% filt 350Hz
fltEMG7=funcNotchFilter(fltEMG6,500,settings.fs);
% filter signal EMG with HP 60Hz
EMGDataFiltered=funcHighPassFilter(fltEMG7,4,60,settings.fs);
% EMGDataFiltered=funcLowPassFilter(fltEMG8,4,2000,settings.fs);
% Calculate power
% pwrEMG=sqrt(fltEMG.*fltEMG);
% EMGDataFiltered=abs(EMGDataFiltered);
EMGDataPowered=smooth((EMGDataFiltered.*EMGDataFiltered),1000);
EMGDataPowered=EMGDataPowered';
figure(21)
plot(EMGDataRaw);
hold on;
plot(EMGDataFiltered);

% ȡ����
% EMGDataEnvelope=envelope(EMGDataFiltered,100,'rms');
% EMGDataEnvelope=smooth(EMGDataEnvelope,99);
% EMGDataEnvelope=EMGDataEnvelope';
% figure(22)
% plot(EMGDataFiltered(1,280001:350000));
% hold on;
% plot(EMGDataEnvelope(1,280001:350000));

% M3��һ�μ��紦��ǰ��
figure(24)
plot(EMGDataRaw(1,280001:350000),'r');
hold on;
plot(EMGDataFiltered(1,280001:350000),'k','LineWidth',1.5);
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 8);

% M3��һ�μ���
figure(23)
plot(abs(EMGDataFiltered(1,280001:350000)),'r');
hold on;
plot(EMGDataPowered(1,280001:350000)/70,'k','LineWidth',1.5);
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 8);


figure(24)
plot(EMGDataPowered);
hold on;
plot(MoveData_Filter);

%% ��EMG�źŽ���ȥ����Ư��
% EMGDataDetrend=detrend(EMGDataPowered,'constant');
% figure(25)
% plot(EMGDataDetrend);
% hold on;
% plot(EMGDataPowered);
EMGDataDetrend=EMGDataPowered;

%% ������Ԥ����
% CSP��CAR��common average referencing filter ���ο���ֵ�˲�
sumall=0;
for i=1:4
    sumall=sumall+NeuroDataRead(i,:);
end
channel_mean=sumall./4;
for i=1:4
    NeuroDataFilter(i,:)=NeuroDataRead(i,:)-channel_mean;
end
% Savitzky-Golay �˲���
for i=1:4
    NeuroDataFilter(i,:)=sgolayfilt(NeuroDataFilter(i,:),3,49);
end

figure(31);
plot(NeuroDataRead(1,:));
hold on;
plot(NeuroDataFilter(1,:));
figure(30);
subplot(2,1,1);
plot(abs(EMGDataFiltered(1,280001:350000)),'r');
hold on;
plot(EMGDataPowered(1,280001:350000)/70,'Color',[0,0,1],'LineWidth',1.5);



%% ��ȡtrial������ѹ���ź���EMG��ͬ����
pick_index=[];
baseline_lever=mean(MoveData_Filter(1,size(MoveData_Filter,2)-1000:end));
threshold_lever=baseline_lever+50;
baseline_emg=funcFindNMin(EMGDataDetrend,100000);
threshold_emg=baseline_emg+200000;
% threshold_emg=baseline_emg+8000;
front=1.5;        %��λs
back=1;         %��λs
[pick_index]=funcPickTrialSameTime(MoveData_Filter,EMGDataDetrend,NeuroDataFilter,threshold_lever,threshold_emg,settings.fs,front,back);

%% �������ȫ��trial
% ѹ�˵�means+-std
MoveDataALL=[];     % ����ƴ��
for i=1:size(pick_index,2)
    MoveDataALL=[MoveDataALL;MoveData_Filter(pick_index(1,i):pick_index(2,i))];
end
MoveMean=mean(MoveDataALL);
MoveStd=std(MoveDataALL);
figure(41);
funcDrawSdorse(1:1:size(MoveMean,2),MoveMean,MoveStd,'b','b',0.5,1);
% �����means+-std
EMGDataALL=[];
for i=1:size(pick_index,2)
    EMGDataALL=[EMGDataALL;funcNormalization(EMGDataDetrend(pick_index(1,i):pick_index(2,i)),0,1)];
end
EMGMean=mean(EMGDataALL);
EMGStd=std(EMGDataALL);
figure(42);
funcDrawSdorse(1:1:size(EMGMean,2),EMGMean,EMGStd,'b','b',0.5,1);
% �Ե��means+-std
NeuroDataALL=[];
for i=1:size(pick_index,2)
    NeuroDataALL=[NeuroDataALL;NeuroDataFilter(4,pick_index(1,i):pick_index(2,i))];
end
NeuroMean=mean(NeuroDataALL);
NeuroStd=std(NeuroDataALL);
figure(43);
funcDrawSdorse(1:1:size(NeuroMean,2),NeuroMean,NeuroStd,'b','b',0.5,1);

% ------���Ƽ����Ե缰����ƴ��ȫ������---------------
gcf=figure(44);
% set(gcf,'unit','inches','width',8,'height',12);
subplot(3,1,1);
plot(EMGDataRaw(1,280001:350000),'Color',[2,135,209]/255);
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 10);
set(gca,'xticklabel',[0 5 10 15 20 25 30 35]);
title('(a)');
% xlabel('Time(s)');
ylabel('Amplitude(uV)');


subplot(3,1,2);
plot(abs(EMGDataFiltered(1,280001:350000)),'Color',[21,101,192]/255);
hold on;
plot(EMGDataPowered(1,280001:350000)/70,'Color',[28,28,28]/255,'LineWidth',1.5);
% hold on;
% plot(MoveData_Filter(1,280001:350000)*20,'Color',[200,0,0]/255)
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 10);
set(gca,'xticklabel',[0 5 10 15 20 25 30 35]);
% set(gca,'Position',[0.2 0.75 0.65 0.175]);
% title('(a)The EMG data and its scaled power');
title('(b)');
% xlabel('Time(s)');
ylabel('Amplitude(uV)');

subplot(3,1,3);
plot(NeuroDataFilter(4,280001:350000),'Color',[40,53,147]/255);
set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 10);
set(gca,'xticklabel',[0 5 10 15 20 25 30 35]);
% set(gca,'Position',[0.2 0.45 0.65 0.175]);
% title('(b)The filtered LFP signals on channel 4');
title('(c)');
xlabel('Time(s)');
ylabel('Amplitude(uV)');

% subplot(3,2,5);
% funcDrawSdorse(1:1:size(NeuroMean(1:6000),2),NeuroMean(1:6000),NeuroStd(1:6000),[0,0,205]/255,[176,196,222]/255,0.5,1);
% set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 10);
% set(gca,'xticklabel',[0 1 2 3]);
% % set(gca,'Position',[0.2 0.15 0.295 0.175]);
% % title('(c)The mean��SD of LFP on channel 4');
% title('(c)');
% xlabel('Time(s)');
% ylabel('Amplitude(uV)');
% 
% subplot(3,2,6);
% funcDrawSdorse(1:1:size(EMGMean(1:6000),2),EMGMean(1:6000),EMGStd(1:6000),[0,0,205]/255,[176,196,222]/255,0.5,1);
% set(gca, 'Fontname', 'Times New Roman', 'Fontsize', 10);
% set(gca,'xticklabel',[0 1 2 3]);
% % set(gca,'Position',[0.555 0.15 0.295 0.175]);
% % title('(d)The mean��SD of powered EMG');
% title('(d)');
% xlabel('Time(s)');
% ylabel('Value');
% 
% print(gcf,'MySavedPlot','-dtiffn')
% -----------------------------------------------

%% ����ƴ��ȫ��trial
MoveDataPicked=[];
NeuroDataPicked=[];
EMGDataPicked=[];
RawNeuroDataPicked=[];
RawEMGDataPicked=[];
for i=1:size(pick_index,2)
    MoveDataPicked=[MoveDataPicked,MoveData_Filter(:,pick_index(1,i):pick_index(2,i))];
    NeuroDataPicked=[NeuroDataPicked,NeuroDataFilter(:,pick_index(1,i):pick_index(2,i))];
    EMGDataPicked=[EMGDataPicked,EMGDataDetrend(:,pick_index(1,i):pick_index(2,i))];
end
figure(51)
plot(MoveDataPicked);
hold on;
plot(EMGDataPicked);
%M2��һ�����ź�
% figure(52)
% subplot(2,1,1);
% plot(NeuroDataPicked(1,361001:391000));
% subplot(2,1,2);
% plot(EMGDataPicked(1,361001:391000)-1600);


%% EMG�ź�FFT
% EMG_fft=funcFFT(EMGDataPicked,settings.fs);

%% EMG��ѹ��ͳһ��һ��
MoveDataNor=funcNormalization(MoveDataPicked,0,300);  %��һ��0-100֮��
EMGDataNor=funcNormalization(EMGDataPicked,0,300);
figure(61)
plot(MoveDataNor);
hold on;
plot(EMGDataNor);

figure(62)
plot(EMGDataNor);
hold on;
plot(NeuroDataPicked(1,:));

%% ���˶��źź�EMG�źŻ�������
BinWidth=50;
MoveDataBinned=funcBinData(MoveDataNor,BinWidth);
EMGDataBinned=funcBinData(EMGDataNor,BinWidth);

%%
MoveData=MoveDataBinned;
EMGData=EMGDataBinned;

%% �׻�ת��
R2=funcCrossCorrelation(NeuroDataPicked);
figure(71)
funcDrawImagesc(R2,'Cross-Correlation between different Channels','Channel','Channel','Times New Roman',8);
[NeuroData, Aw, means] = funcWhiteningTransform(NeuroDataPicked);
NeuroData=real(NeuroData);
R3=funcCrossCorrelation(NeuroData);
figure(72);
funcDrawImagesc(R3,'Cross-Correlation between different Sources','Channel','Channel','Times New Roman',8);

%% ���㹦���ײ�����
for i=1:size(NeuroData,1)
%     [pxx,f] =  pwelch(NeuroData(i,:),100,50,256,settings.fs);      %���ƹ�����
%     plot(f,10*log10(pxx))
    
    [s,f,t,p]=funcSpectroEstimator(NeuroData,i,settings.fs,512);
    
    figure(i+80);
    funcDrawSpectrom(zscore(p,1,2),MoveData,'Times New Roman',8)
    figure(i+90);
    funcDrawSpectrom(zscore(p,1,2),EMGData,'Times New Roman',8)
    
    % ======== ��������źŽ���ƴ�� ============================
    trial_interval=floor(size(p,2)/size(pick_index,2));
    NeuroDataSpectroALL=zeros(size(p,1),trial_interval);
    for j=1:size(pick_index,2)
        if(trial_interval*j<size(p,2))
            NeuroDataSpectroALL=NeuroDataSpectroALL+p(:,(j-1)*trial_interval+1:trial_interval*j);
        end
    end
    NeuroDataSpectroALL=zscore(NeuroDataSpectroALL/size(pick_index,2),1,2);
    %---------datafile20190510M2������β���ݵ���-------------
    for nn=1:10
        NeuroDataSpectroALL(:,nn)=NeuroDataSpectroALL(:,11);
    end
%     for mm=95:100
%         NeuroDataALL(:,mm)=NeuroDataALL(:,end-6);
%     end
    %-------------------------------------------------------
    MoveMeanBinned=funcBinData(MoveMean,BinWidth);
    EMGMeanBinned=funcBinData(EMGMean,BinWidth);
    figure(i+190);
    funcDrawSpectrom(NeuroDataSpectroALL(:,1:end-1),MoveMeanBinned(:,1:end),'Times New Roman',8);
    if(i==3)
        save('NeuroDataSpectroALL3.mat','NeuroDataSpectroALL');
    end
    figure(i+200);
    funcDrawSpectrom(NeuroDataSpectroALL(:,1:end-1),EMGMeanBinned(:,1:end)*200,'Times New Roman',8);
    hold on;
    plot(MoveMeanBinned(:,1:end));
    % ========================================================
    
    NeuroFeatureS(:,:,i)=s;   %ԭʼ�ź�
    NeuroFeatureF(:,:,i)=f;
    NeuroFeatureT(:,:,i)=t;
    NeuroFeatureP(:,:,i)=p;
end

% NeuroFeatureExtract=NeuroFeatureP;


%% ɾ��300Hz�����ź�
for j=1:size(NeuroFeatureF,1)
    if NeuroFeatureF(j,1,1)>=300
        index_delete=j;
        break;
    end
end
index_delete=index_delete+1;        % ��������һ������300��ֵ
NeuroFeatureS(index_delete:end,:,:)=[];
NeuroFeatureF(index_delete:end,:,:)=[];
NeuroFeatureT(index_delete:end,:,:)=[];
NeuroFeatureP(index_delete:end,:,:)=[];


%% ������ȡ���ϲ�-��ά��
% feature_box=[0 4 8 12 30 50 120 250];   %�ģ�1��3Hz�����ȣ�4��7Hz��������8��13Hz�����£�14��30Hz��
feature_box=[0 8 30 50 70 100 150 200];
%feature_box=[0 48 120 200 280 400 600 800];
NFeatureExtract=[];
NeuroFeatureExtract=[];
for channel=1:4
    for i=1:size(feature_box,2)-1
       index=find(NeuroFeatureF(:,1,channel)>=feature_box(i) & NeuroFeatureF(:,1,channel)<feature_box(i+1));
       NFeatureExtract=[NFeatureExtract;sum(NeuroFeatureP(index,:,channel),1)];
    end
    NeuroFeatureExtract(:,:,channel)=NFeatureExtract;
    NFeatureExtract=[];
end


%% ѹ�������źŵ�����Է���
% ����Է�������ͨ���ۼӵĲ�ͬƵ��
% XN=zeros(size(NeuroFeatureExtract,1),size(NeuroFeatureExtract,2));
% for i=1:4
%     temp=reshape(NeuroFeatureExtract(:,:,i),size(NeuroFeatureExtract,1),size(NeuroFeatureExtract,2));
%     XN=XN+temp;
% end
% Y=MoveData;
% for j=1:size(NeuroFeatureExtract,1)
%     X=XN(j,:);
%     [corr_ratio_move(j),corr_p_value_move(j)]=funcCorrelation(X,Y);
% end
% figure(115);
% bar(corr_ratio_move,'k');
% set(gca,'xticklabel',{'lfLFP','��','��','��','��1','��2','��3','bhfLFP'});
% ylabel('Correlation');

% ����Է�����ͬͨ������ͬƵ��
% for i=1:4
%     XN=NeuroFeatureExtract(:,:,i);
%     Y=MoveData;
%     for j=1:size(NeuroFeatureExtract,1)
%         X=XN(j,:);
%         corr_ratio(i,j)=funcCorrelation(X,Y);
%         
%     end
%     figure(100+i);
%     funcDrawBar(corr_ratio(i,:),'Correlation between each frequency band and lever pressure','Frequency band','Correlation','Times New Roman',8);
% end

%% EMG�����źŵ�����Է���
% ����Է�������ͨ���ۼӵĲ�ͬƵ��
XN=zeros(size(NeuroFeatureExtract,1),size(NeuroFeatureExtract,2));
for i=1:4
    temp=reshape(NeuroFeatureExtract(:,:,i),size(NeuroFeatureExtract,1),size(NeuroFeatureExtract,2));
    XN=XN+temp;
end
Y=EMGData;
for j=1:size(NeuroFeatureExtract,1)
    X=XN(j,:);
    [corr_ratio_emg(j),corr_p_value_emg(j)]=funcCorrelation(X,Y);
end
figure(115);
bar(corr_ratio_emg,'k');
set(gca,'xticklabel',{'lfLFP','��','��1L','��1H','��2L','��2H','��3L','��3H'});
ylabel('Correlation');

% ����Է�����ͬͨ������ͬƵ��
% for i=1:4
%     XN=NeuroFeatureExtract(:,:,i);
%     Y=EMGData;
%     for j=1:size(NeuroFeatureExtract,1)
%         X=XN(j,:);
%         corr_ratio(i,j)=funcCorrelation(X,Y);
%         
%     end
%     figure(110+i);
%     funcDrawBar(corr_ratio(i,:),'Correlation between each frequency band and emg','Frequency band','Correlation','Times New Roman',8);
% end

% ------���ƹ�����ͼ�����ϵ����ͼ--------------------------
load('NeuroDataSpectroALL3.mat');
gcf130=figure(130);
sub1=subplot(2,9,[1,2,3,4]);
fImagsc1=funcDrawImagesc(R2,'(a)','Channel','Channel','Times New Roman',10);
% set(gca,'FontWeight','bold');
%Cross-Correlation between different Channels
colormap(sub1,parula);

sub2=subplot(2,9,[6,7,8,9]);
fImagsc12=funcDrawImagesc(R3,'(b)','Channel','Channel','Times New Roman',10);
% set(gca,'FontWeight','bold');
colorbar('position',[0.92 0.58 0.02 0.345])
%Cross-Correlation between different Sources
colormap(sub2,parula);

sub3=subplot(2,9,[10,11,12,13]);
funcDrawSpectrom(NeuroDataSpectroALL(1:200,1:end-1),EMGMeanBinned(:,1:end)*200,'Times New Roman',10);
% set(gca,'FontWeight','bold');
set(gca,'xticklabel',[0.5 1 1.5 2 2.5 3]);
colormap(sub3,parula);
% title('(c)Average spectrgram and EMG');
title('(c)');
xlabel('Time(s)');
ylabel('Frequency(Hz)');

subplot(2,9,[15,16,17,18]);
% [0.1986,0.1365,0.2225,0.3032,0.3200,0.3109,0.3210]
funcDrawBar(corr_ratio_emg,'(d)','Frequency band','Correlation','Times New Roman',10);
% set(gca,'FontWeight','bold');
% Correlation between frequency and EMG
set(gca,'xticklabel',{'lfLFP','��','��1','L��2','H��2','L��3','H��3'});
print(gcf130,'fig4','-r600','-dtiffn')
% -------------------------------------------------------


%% ����ƴ�����ݼ�
sample_number=size(EMGData,2);     %������
LFP1=NeuroFeatureExtract(:,1:sample_number*1);
LFP2=NeuroFeatureExtract(:,sample_number*1+1:sample_number*2);
LFP3=NeuroFeatureExtract(:,sample_number*2+1:sample_number*3);
LFP4=NeuroFeatureExtract(:,sample_number*3+1:sample_number*4);
NeuroFeature=[LFP1;LFP2;LFP3;LFP4]; %����

%% �������ݼ�
csvwrite('MoveData.csv',MoveData);
csvwrite('EMGData.csv',EMGData);
% csvwrite('NeuroFeatureF.csv',NeuroFeatureF);
% csvwrite('NeuroFeatureT.csv',NeuroFeatureT);
% csvwrite('NeuroFeatureP.csv',NeuroFeatureP);
csvwrite('NeuroFeature.csv',NeuroFeature);

