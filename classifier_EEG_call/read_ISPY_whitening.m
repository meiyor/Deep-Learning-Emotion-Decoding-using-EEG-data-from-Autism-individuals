function read_ISPY_whitening(subject_beginning,sel_conv_read,time_range,ch,map_l)
load('chanloc_ref.mat','ref_chanlocs');
t=linspace(-200,1550,875);
pos1=max(find(t<=time_range(1)));
pos2=min(find(t>=time_range(2)));
size_d=length([pos1:1:pos2]);
%% ISPY
path='S:\JMM_EEG_data_copy\ISPY\EEG_Recording';
%[dat_TD,dat_ASD]=behavioral_eval('ISPY_n','C:\Users\jmayortorres\Documents\EEG_main_folder\data_csvs\ISPY_DATA_LABELS_2017-11-22_1913_TD_ASD_2.csv','C:\Users\jmayortorres\Documents\EEG_main_folder\data_csvs\ISPY_DATA_LABELS_2017-11-22_1530_Performances.csv',96,7,2,6,1);
%[dat_TD,dat_ASD]=behavioral_eval('ISPY_n','C:\Users\jmayortorres\Documents\EEG_main_folder\data_csvs\ISPY_DATA_LABELS_2017-11-22_1913_TD_ASD_2.csv','C:\Users\jmayortorres\Documents\EEG_main_folder\data_csvs\ISPY_DATA_LABELS_2018-01-15_1544_Performances.csv',96,7,2,6,1);
close all;
close all hidden;
A_dir=dir(path);
n_count=1;
happy=zeros([1 875]);
sad=zeros([1 875]);
angry=zeros([1 875]);
fear=zeros([1 875]);
happy_c=zeros([1 30]);
sad_c=zeros([1 30]);
angry_c=zeros([1 30]);
fear_c=zeros([1 30]);
happy_td=zeros([1 875]);
sad_td=zeros([1 875]);
angry_td=zeros([1 875]);
fear_td=zeros([1 875]);
happy_c_td=zeros([1 30]);
sad_c_td=zeros([1 30]);
angry_c_td=zeros([1 30]);
fear_c_td=zeros([1 30]);
happy_asd=zeros([1 875]);
sad_asd=zeros([1 875]);
angry_asd=zeros([1 875]);
fear_asd=zeros([1 875]);
happy_c_asd=zeros([1 30]);
sad_c_asd=zeros([1 30]);
angry_c_asd=zeros([1 30]);
fear_c_asd=zeros([1 30]);
happy_v=[];
sad_v=[];
angry_v=[];
fear_v=[];
%n_TD=1;
%n_ASD=1;
for k=3+subject_beginning(1):length(A_dir)
        B_dir=dir([path '/' A_dir(k).name]);
        for p=3:length(B_dir)
            if length(B_dir(p).name)==13 && all(B_dir(p).name=='DANVA_res.mat')
               temp_str=load([path '/' A_dir(k).name '/' B_dir(p).name]);
               if (length(temp_str.EEG_val_happy)~=0)
                    EEG_DATA{n_count}=temp_str;
                    S1=size(EEG_DATA{n_count}.EEG_val_happy{1}.data);
                    if (length(EEG_DATA{n_count}.EEG_val_happy{1}.chanlocs)~=30)
                        dat_happy=EEG_DATA{n_count}.EEG_val_happy{1};
                        dat_sad=EEG_DATA{n_count}.EEG_val_sad{1};
                        dat_angry=EEG_DATA{n_count}.EEG_val_angry{1}; 
                        dat_fear=EEG_DATA{n_count}.EEG_val_fear{1}; 
                        dat_happy.data=find_realpos(dat_happy.chanlocs,ref_chanlocs,dat_happy.data);
                        dat_sad.data=find_realpos(dat_sad.chanlocs,ref_chanlocs,dat_sad.data);
                        dat_angry.data=find_realpos(dat_angry.chanlocs,ref_chanlocs,dat_angry.data);
                        dat_fear.data=find_realpos(dat_fear.chanlocs,ref_chanlocs,dat_fear.data);
                        %% reverse the assigment for doing quicker and efficiently
                        EEG_DATA{n_count}.EEG_val_happy{1}.data(:,:,:)=dat_happy.data;
                        EEG_DATA{n_count}.EEG_val_sad{1}.data(:,:,:)=dat_sad.data;
                        EEG_DATA{n_count}.EEG_val_angry{1}.data(:,:,:)=dat_angry.data;
                        EEG_DATA{n_count}.EEG_val_fear{1}.data(:,:,:)=dat_fear.data;
                    end;
                    XT_DATA{n_count}=[reshape(permute(EEG_DATA{n_count}.EEG_val_happy{1}.data(1:30,pos1:pos2,:),[2 1 3]),30*size_d,S1(3)) reshape(permute(EEG_DATA{n_count}.EEG_val_sad{1}.data(1:30,pos1:pos2,:),[2 1 3]),30*size_d,S1(3)) reshape(permute(EEG_DATA{n_count}.EEG_val_angry{1}.data(1:30,pos1:pos2,:),[2 1 3]),30*size_d,S1(3)) reshape(permute(EEG_DATA{n_count}.EEG_val_fear{1}.data(1:30,pos1:pos2,:),[2 1 3]),30*size_d,S1(3))];
                    subject_code{n_count}=A_dir(k).name;
                    n_count=n_count+1;
               end;
            end;
        end;
end;
pp=[1:50 52:66];
ASD_index=[1 2 6 11 12 14 17 18 19 20 29 31 32 33 34 36 41 42 43 45 46 49 50 54 56 60 61 62 66];
TD_index=setdiff(pp,ASD_index);
td_i=1;
asd_i=1;
for nn=1:length(pp)
    if length(ch)~=1
      happy=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(ch,:,:)),1)),2)'+happy;
      sad=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(ch,:,:)),1)),2)'+sad;
      angry=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(ch,:,:)),1)),2)'+angry;
      fear=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(ch,:,:)),1)),2)'+fear;
    else
      happy=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(ch,:,:)),2)'+happy;
      sad=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(ch,:,:)),2)'+sad;
      angry=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(ch,:,:)),2)'+angry;
      fear=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(ch,:,:)),2)'+fear; 
    end;
      happy_c=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(1:30,pos1:pos2,:)),2)),2)'+happy_c;
      sad_c=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(1:30,pos1:pos2,:)),2)),2)'+sad_c;
      angry_c=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(1:30,pos1:pos2,:)),2)),2)'+angry_c;
      fear_c=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(1:30,pos1:pos2,:)),2)),2)'+fear_c;
      if any(TD_index==pp(nn))
         if length(ch)~=1
          happy_td=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(ch,:,:)),1)),2)'+happy_td;
          sad_td=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(ch,:,:)),1)),2)'+sad_td;
          angry_td=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(ch,:,:)),1)),2)'+angry_td;
          fear_td=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(ch,:,:)),1)),2)'+fear_td;
         else
             happy_td=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(ch,:,:)),2)'+happy_td;
             sad_td=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(ch,:,:)),2)'+sad_td;
             angry_td=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(ch,:,:)),2)'+angry_td;
             fear_td=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(ch,:,:)),2)'+fear_td; 
         end;
          happy_c_td=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(1:30,pos1:pos2,:)),2)),2)'+happy_c_td;
          sad_c_td=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(1:30,pos1:pos2,:)),2)),2)'+sad_c_td;
          angry_c_td=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(1:30,pos1:pos2,:)),2)),2)'+angry_c_td;
          fear_c_td=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(1:30,pos1:pos2,:)),2)),2)'+fear_c_td;
          happy_TD(td_i)=mean(mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(ch,pos1:pos2,:)),1)),2));
          sad_TD(td_i)=mean(mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(ch,pos1:pos2,:)),1)),2));
          angry_TD(td_i)=mean(mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(ch,pos1:pos2,:)),1)),2));
          fear_TD(td_i)=mean(mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(ch,pos1:pos2,:)),1)),2));
          td_i=td_i+1;
      end;
      if any(ASD_index==pp(nn))
        if length(ch)~=1  
          happy_asd=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(ch,:,:)),1)),2)'+happy_asd;
          sad_asd=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(ch,:,:)),1)),2)'+sad_asd;
          angry_asd=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(ch,:,:)),1)),2)'+angry_asd;
          fear_asd=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(ch,:,:)),1)),2)'+fear_asd;
        else
            happy_asd=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(ch,:,:)),2)'+happy_asd;
            sad_asd=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(ch,:,:)),2)'+sad_asd;
            angry_asd=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(ch,:,:)),2)'+angry_asd;
            fear_asd=mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(ch,:,:)),2)'+fear_asd;  
        end;
          happy_c_asd=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(1:30,pos1:pos2,:)),2)),2)'+happy_c_asd;
          sad_c_asd=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(1:30,pos1:pos2,:)),2)),2)'+sad_c_asd;
          angry_c_asd=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(1:30,pos1:pos2,:)),2)),2)'+angry_c_asd;
          fear_c_asd=mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(1:30,pos1:pos2,:)),2)),2)'+fear_c_asd;
          happy_ASD(asd_i)=mean(mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_happy{1}.data(ch,pos1:pos2,:)),1)),2));
          sad_ASD(asd_i)=mean(mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_sad{1}.data(ch,pos1:pos2,:)),1)),2));
          angry_ASD(asd_i)=mean(mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_angry{1}.data(ch,pos1:pos2,:)),1)),2));
          fear_ASD(asd_i)=mean(mean(squeeze(mean(squeeze(EEG_DATA{pp(nn)}.EEG_val_fear{1}.data(ch,pos1:pos2,:)),1)),2));
          asd_i=asd_i+1;
      end;
%       happy_v(nn,:)=mean(squeeze(EEG_DATA{p(nn)}.EEG_val_happy{1}.data(ch,:,:)),2)';
%       sad_v(nn,:)=mean(squeeze(EEG_DATA{p(nn)}.EEG_val_sad{1}.data(ch,:,:)),2)';
%       angry_v(nn,:)=mean(squeeze(EEG_DATA{p(nn)}.EEG_val_angry{1}.data(ch,:,:)),2)';
%       fear_v(nn,:)=mean(squeeze(EEG_DATA{p(nn)}.EEG_val_fear{1}.data(ch,:,:)),2)';
end;
plot(t,happy./(length(EEG_DATA)-1),'b','LineWidth',4);
hold on;
plot(t,sad./(length(EEG_DATA)-1),'g','LineWidth',4);
plot(t,angry./(length(EEG_DATA)-1),'r','LineWidth',4);
plot(t,fear./(length(EEG_DATA)-1),'k','LineWidth',4);
legend({'Happy','Sad','Angry','Fear'});
grid on;
set(gca,'FontSize',20)
xlabel('Time [ms]');
ylabel('Amplitude [uV]');
figure;
plot(t,happy_td./(36),'b','LineWidth',4);
hold on;
plot(t,sad_td./(36),'g','LineWidth',4);
plot(t,angry_td./(36),'r','LineWidth',4);
plot(t,fear_td./(36),'k','LineWidth',4);
grid on;
hold on;
plot(t,happy_asd./(29),'b:','LineWidth',4);
plot(t,sad_asd./(29),'g:','LineWidth',4);
plot(t,angry_asd./(29),'r:','LineWidth',4);
plot(t,fear_asd./(29),'k:','LineWidth',4);
legend({'Happy TD','Sad TD','Angry TD','Fear TD','Happy ASD','Sad ASD','Angry ASD','Fear ASD'});
grid on;
makeBar(t(pos1:pos2),-3,0.0001);
set(gca,'FontSize',20)
title('TD N=36 ASD N=29 Parietal')
set(gca,'FontSize',20)
xlabel('Time [ms]');
ylabel('Amplitude [uV]');
figure;
subplot(221);
title('Happy')
topoplot(happy_c./(length(EEG_DATA)-1),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',[-1 2]);
set(gca,'FontSize',20)
subplot(222);
title('Sad')
topoplot(sad_c./(length(EEG_DATA)-1),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',[-1 2]);
set(gca,'FontSize',20)
subplot(223);
title('Angry')
topoplot(angry_c./(length(EEG_DATA)-1),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',[-1 2]);
set(gca,'FontSize',20)
subplot(224);
title('Fear')
topoplot(fear_c./(length(EEG_DATA)-1),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',[-1 2]);
set(gca,'FontSize',20)
figure;
subplot(221);
title('Happy TD')
topoplot(happy_c_td./(36),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',map_l);
colorbar
set(gca,'FontSize',20)
subplot(222);
title('Sad TD')
topoplot(sad_c_td./(36),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',map_l);
colorbar
set(gca,'FontSize',20)
subplot(223);
title('Angry TD')
topoplot(angry_c_td./(36),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',map_l);
colorbar
set(gca,'FontSize',20)
subplot(224);
title('Fear TD')
topoplot(fear_c_td./(36),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',map_l);
colorbar
set(gca,'FontSize',20)
figure;
subplot(221);
title('Happy ASD')
topoplot(happy_c_asd./(29),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',map_l);
colorbar
set(gca,'FontSize',20)
subplot(222);
title('Sad ASD')
topoplot(sad_c_asd./(29),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',map_l);
colorbar
set(gca,'FontSize',20)
subplot(223);
title('Angry ASD')
topoplot(angry_c_asd./(29),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',map_l);
colorbar
set(gca,'FontSize',20)
subplot(224);
title('Fear ASD')
topoplot(fear_c_asd./(29),ref_chanlocs,'electrodes','on','electrodes','labels','maplimits',map_l);
colorbar
set(gca,'FontSize',20)
data1=[happy_TD,sad_TD,angry_TD,fear_TD];
data2=[happy_ASD,sad_ASD,angry_ASD,fear_ASD];
RM_ANOVA(data1,data2)
A=1;
%% whitening image creation
%% First TD
ntd=1;
for k=1:length(pp)
   if any(TD_index==pp(k))
    XDATA=prdataset(double(XT_DATA{pp(k)}'),[ones([1 12]) 2*ones([1 12]) 3*ones([1 12]) 4*ones([1 12])]');
     v_pos=[1:1:48];
     for l_pos=1:size(XDATA,1)
            Xtrain{l_pos}=XDATA(find(v_pos~=l_pos),:);
            Xtest{l_pos}=XDATA(find(v_pos==l_pos),:);
             %% PCA/ZCA whitening before sending it to Tensorflow
             %if sel_conv_read==1 && l_pos==1 && ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}],'dir')
             if sel_conv_read==1 && l_pos==1
                    for p_c=1:size(Xtrain{l_pos}.data,1)
                        avg=mean(reshape(Xtrain{l_pos}.data(p_c,:),size_d,30),1);     % Compute the mean pixel intensity value separately for each patch. 
                        X=reshape(Xtrain{l_pos}.data(p_c,:),size_d,30)-repmat(avg,size(reshape(Xtrain{l_pos}.data(p_c,:),size_d,30),1), 1);
                        sigma_tr = X*X'/size(X,2);
                        [U_tr,S_tr,V_tr]=svd(sigma_tr);
                        xRot_tr=U_tr'*X;          
                        xTilde_tr=U_tr(:,1:10)'*X;
                        xPCAwhite_tr=diag(1./sqrt(diag(S_tr)+0.1))*U_tr'*X;
                        xZCAwhite_tr = U_tr * diag(1./sqrt(diag(S_tr) + 0.1))*U_tr'*X;
                        Xtrain_t{l_pos}(p_c,:)=reshape(xZCAwhite_tr,1,30*size_d);
                    end;
                    %xZCAwhite_tr
                    for q_c=1:size(Xtest{l_pos}.data,1)
                        avg=mean(reshape(Xtest{l_pos}.data(q_c,:),size_d,30),1);     % Compute the mean pixel intensity value separately for each patch. 
                        Xt=reshape(Xtest{l_pos}.data(q_c,:),size_d,30)-repmat(avg,size(reshape(Xtest{l_pos}.data(q_c,:),size_d,30),1), 1);
                        sigma_t = Xt*Xt'/size(Xt,2);
                        [U_t,S_t,V_tr]=svd(sigma_t);
                        xRot_t=U_t'*Xt;          
                        xTilde_t=U_t(:,1:10)'*Xt;
                        xPCAwhite_t=diag(1./sqrt(diag(S_t)+0.1))*U_t'*Xt;
                        xZCAwhite_t = U_t * diag(1./sqrt(diag(S_t) + 0.1))*U_t'*Xt;
                        Xtest_t{l_pos}(q_c,:)=reshape(xZCAwhite_t,1,30*size_d);
                    end;
                   Ximag(ntd,:,:)=[[abs(Xtest_t{l_pos})] ; [abs(Xtrain_t{l_pos})]];
                   ntd=ntd+1;
                   %% process the whitening images per subject
                    %% add the condition for folder existance
                   % if ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}],'dir')
                   %     mkdir(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}]);
                   %     csvwrite(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)} '\data_sub_' subject_code{vector_pos(k)} '.csv'],[[abs(Xtest_t{l_pos})  getlabels(Xtest{l_pos})] ; [abs(Xtrain_t{l_pos})  getlabels(Xtrain{l_pos})]]);
                    %if ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}],'dir')
                    %    mkdir(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}]);
                    %    csvwrite(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)} '\data_sub_' subject_code{vector_pos(k)} '.csv'],[[abs(Xtest_t{l_pos})  getlabels(Xtest{l_pos})] ; [abs(Xtrain_t{l_pos})  getlabels(Xtrain{l_pos})]]); 
                  % end
               end;
        end;
   end;
end;
%% process the image whitening average
Xaveimag_happy=zeros([1 30]);
Xaveimag_sad=zeros([1 30]);
Xaveimag_angry=zeros([1 30]);
Xaveimag_fear=zeros([1 30]);
for n=1:size(Ximag,1)
    temp=max(reshape(mean(squeeze(Ximag(n,1:12,:)),1),size_d,30));
    temp(find(temp==0))=0.24;
    Xaveimag_happy=Xaveimag_happy+temp;
    temp=max(reshape(mean(squeeze(Ximag(n,13:24,:)),1),size_d,30));
    temp(find(temp==0))=0.24;
    Xaveimag_sad=Xaveimag_sad+temp;
    temp=max(reshape(mean(squeeze(Ximag(n,25:36,:)),1),size_d,30));
    temp(find(temp==0))=0.24;
    Xaveimag_angry=Xaveimag_angry+temp;
    temp=max(reshape(mean(squeeze(Ximag(n,37:48,:)),1),size_d,30));
    temp(find(temp==0))=0.24;
    Xaveimag_fear=Xaveimag_fear+temp;
end;
A=1;
%% Second ASD
nasd=1;
for k=1:length(pp)
   if any(ASD_index==pp(k))
    XDATA=prdataset(double(XT_DATA{pp(k)}'),[ones([1 12]) 2*ones([1 12]) 3*ones([1 12]) 4*ones([1 12])]');
     v_pos=[1:1:48];
     for l_pos=1:size(XDATA,1)
            Xtrain{l_pos}=XDATA(find(v_pos~=l_pos),:);
            Xtest{l_pos}=XDATA(find(v_pos==l_pos),:);
             %% PCA/ZCA whitening before sending it to Tensorflow
             %if sel_conv_read==1 && l_pos==1 && ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}],'dir')
             if sel_conv_read==1 && l_pos==1
                    for p_c=1:size(Xtrain{l_pos}.data,1)
                        avg=mean(reshape(Xtrain{l_pos}.data(p_c,:),size_d,30),1);     % Compute the mean pixel intensity value separately for each patch. 
                        X=reshape(Xtrain{l_pos}.data(p_c,:),size_d,30)-repmat(avg,size(reshape(Xtrain{l_pos}.data(p_c,:),size_d,30),1), 1);
                        sigma_tr = X*X'/size(X,2);
                        [U_tr,S_tr,V_tr]=svd(sigma_tr);
                        xRot_tr=U_tr'*X;          
                        xTilde_tr=U_tr(:,1:10)'*X;
                        xPCAwhite_tr=diag(1./sqrt(diag(S_tr)+0.1))*U_tr'*X;
                        xZCAwhite_tr = U_tr * diag(1./sqrt(diag(S_tr) + 0.1))*U_tr'*X;
                        Xtrain_t{l_pos}(p_c,:)=reshape(xZCAwhite_tr,1,30*size_d);
                    end;
                    %xZCAwhite_tr
                    for q_c=1:size(Xtest{l_pos}.data,1)
                        avg=mean(reshape(Xtest{l_pos}.data(q_c,:),size_d,30),1);     % Compute the mean pixel intensity value separately for each patch. 
                        Xt=reshape(Xtest{l_pos}.data(q_c,:),size_d,30)-repmat(avg,size(reshape(Xtest{l_pos}.data(q_c,:),size_d,30),1), 1);
                        sigma_t = Xt*Xt'/size(Xt,2);
                        [U_t,S_t,V_tr]=svd(sigma_t);
                        xRot_t=U_t'*Xt;          
                        xTilde_t=U_t(:,1:10)'*Xt;
                        xPCAwhite_t=diag(1./sqrt(diag(S_t)+0.1))*U_t'*Xt;
                        xZCAwhite_t = U_t * diag(1./sqrt(diag(S_t) + 0.1))*U_t'*Xt;
                        Xtest_t{l_pos}(q_c,:)=reshape(xZCAwhite_t,1,30*size_d);
                    end;
                    Ximag_n(nasd,:,:)=[[abs(Xtest_t{l_pos})] ; [abs(Xtrain_t{l_pos})]];
                    nasd=nasd+1;
                    %% add the condition for folder existance
                   % if ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}],'dir')
                   %     mkdir(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}]);
                   %     csvwrite(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)} '\data_sub_' subject_code{vector_pos(k)} '.csv'],[[abs(Xtest_t{l_pos})  getlabels(Xtest{l_pos})] ; [abs(Xtrain_t{l_pos})  getlabels(Xtrain{l_pos})]]);
                   % if ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}],'dir')
                   %     mkdir(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}]);
                   %     csvwrite(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)} '\data_sub_' subject_code{vector_pos(k)} '.csv'],[[abs(Xtest_t{l_pos})  getlabels(Xtest{l_pos})] ; [abs(Xtrain_t{l_pos})  getlabels(Xtrain{l_pos})]]); 
                   %end
               end;
        end;
   end;
end;
%% process the image whitening average
Xaveimag_happy_n=zeros([1 30]);
Xaveimag_sad_n=zeros([1 30]);
Xaveimag_angry_n=zeros([1 30]);
Xaveimag_fear_n=zeros([1 30]);
for n=1:size(Ximag_n,1)
    temp=max(reshape(mean(squeeze(Ximag_n(n,1:12,:)),1),size_d,30));
    temp(find(temp==0))=0.24;
    Xaveimag_happy_n=Xaveimag_happy_n+temp;
    temp=max(reshape(mean(squeeze(Ximag_n(n,13:24,:)),1),size_d,30));
    temp(find(temp==0))=0.24;
    Xaveimag_sad_n=Xaveimag_sad_n+temp;
    temp=max(reshape(mean(squeeze(Ximag_n(n,25:36,:)),1),size_d,30));
    temp(find(temp==0))=0.24;
    Xaveimag_angry_n=Xaveimag_angry_n+temp;
    temp=max(reshape(mean(squeeze(Ximag_n(n,37:48,:)),1),size_d,30));
    temp(find(temp==0))=0.24;
    Xaveimag_fear_n=Xaveimag_fear_n+temp;
end;
A=1;
%% this is only to read the remaining subjects we have to evaluate the TD and ASD subjects separately in a another piece of code
vector_pos=[21 24 25 30 31 33 35 36 41:50 52:66]
for k=1:length(vector_pos)
    XDATA=prdataset(double(XT_DATA{vector_pos(k)}'),[ones([1 12]) 2*ones([1 12]) 3*ones([1 12]) 4*ones([1 12])]');
     v_pos=[1:1:48];
     for l_pos=1:size(XDATA,1)
            Xtrain{l_pos}=XDATA(find(v_pos~=l_pos),:);
            Xtest{l_pos}=XDATA(find(v_pos==l_pos),:);
             %% PCA/ZCA whitening before sending it to Tensorflow
             %if sel_conv_read==1 && l_pos==1 && ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}],'dir')
             if sel_conv_read==1 && l_pos==1 && ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}],'dir')
                    for p_c=1:size(Xtrain{l_pos}.data,1)
                        avg=mean(reshape(Xtrain{l_pos}.data(p_c,:),size_d,30),1);     % Compute the mean pixel intensity value separately for each patch. 
                        X=reshape(Xtrain{l_pos}.data(p_c,:),size_d,30)-repmat(avg,size(reshape(Xtrain{l_pos}.data(p_c,:),size_d,30),1), 1);
                        sigma_tr = X*X'/size(X,2);
                        [U_tr,S_tr,V_tr]=svd(sigma_tr);
                        xRot_tr=U_tr'*X;          
                        xTilde_tr=U_tr(:,1:10)'*X;
                        xPCAwhite_tr=diag(1./sqrt(diag(S_tr)+0.1))*U_tr'*X;
                        xZCAwhite_tr = U_tr * diag(1./sqrt(diag(S_tr) + 0.1))*U_tr'*X;
                        Xtrain_t{l_pos}(p_c,:)=reshape(xZCAwhite_tr,1,30*size_d);
                    end;
                    %xZCAwhite_tr
                    for q_c=1:size(Xtest{l_pos}.data,1)
                        avg=mean(reshape(Xtest{l_pos}.data(q_c,:),size_d,30),1);     % Compute the mean pixel intensity value separately for each patch. 
                        Xt=reshape(Xtest{l_pos}.data(q_c,:),size_d,30)-repmat(avg,size(reshape(Xtest{l_pos}.data(q_c,:),size_d,30),1), 1);
                        sigma_t = Xt*Xt'/size(Xt,2);
                        [U_t,S_t,V_tr]=svd(sigma_t);
                        xRot_t=U_t'*Xt;          
                        xTilde_t=U_t(:,1:10)'*Xt;
                        xPCAwhite_t=diag(1./sqrt(diag(S_t)+0.1))*U_t'*Xt;
                        xZCAwhite_t = U_t * diag(1./sqrt(diag(S_t) + 0.1))*U_t'*Xt;
                        Xtest_t{l_pos}(q_c,:)=reshape(xZCAwhite_t,1,30*size_d);
                    end;
                    %% add the condition for folder existance
                   % if ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}],'dir')
                   %     mkdir(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}]);
                   %     csvwrite(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)} '\data_sub_' subject_code{vector_pos(k)} '.csv'],[[abs(Xtest_t{l_pos})  getlabels(Xtest{l_pos})] ; [abs(Xtrain_t{l_pos})  getlabels(Xtrain{l_pos})]]);
                    if ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}],'dir')
                        mkdir(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}]);
                        csvwrite(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)} '\data_sub_' subject_code{vector_pos(k)} '.csv'],[[abs(Xtest_t{l_pos})  getlabels(Xtest{l_pos})] ; [abs(Xtrain_t{l_pos})  getlabels(Xtrain{l_pos})]]); 
                   end
               end;
        end;
end;
A=1;


