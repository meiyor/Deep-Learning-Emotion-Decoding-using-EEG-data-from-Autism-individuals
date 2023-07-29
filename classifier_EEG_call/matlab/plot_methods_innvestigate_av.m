function plot_methods_innvestigate_av(path,sel)
close all
methods={'Gradient','SmoothGrad','DeconvNet','GuidedBackprop','PatterNet','PatternAttribution','DeepTaylor','Input * Gradient','Integrated Gradients','LRP-epsilon','LRP-Z','LRP-APresetflat','LRP-BPresetflat'};
A_dir=dir(path);
if sel==1
t=linspace(0,1500,752);
n=1;
data_average=zeros(13,752,30);
data_happy=zeros(13,752,30);
data_sad=zeros(13,752,30);
data_angry=zeros(13,752,30);
data_fear=zeros(13,752,30);
data_innv=zeros(48,48,13,6);
n=1;
fmt=repmat('%d',1,1); 
for k=3:length(A_dir) %% subjects
    data_average_trials=zeros(13,752,30);
    data_happy_trials=zeros(13,752,30);
    data_sad_trials=zeros(13,752,30);
    data_angry_trials=zeros(13,752,30);
    data_fear_trials=zeros(13,752,30);
    for p=1:48 %% trials
        for i=1:13 %% methods
                [path '/' A_dir(k).name '/method_' num2str(p) '_' num2str(i) '.txt']
                fid=fopen([path '/' A_dir(k).name '/method_' num2str(p) '_' num2str(i) '.txt'],'r');
                D = textscan(fid,fmt,'Delimiter',',','CollectOutput',1);
                data=double(reshape(D{1},30,752))';
                [Xwh,mu,invM]=whiten(rescale(data,-1,1),0.001);
                data=rescale(data,-1,1)*invM;
                fclose(fid);
            %    data=csvread([path '/' A_dir(k).name '/method_' num2str(p) '_' num2str(i) '.txt'],0,0);
               % fid=fopen([path '/' A_dir(k).name '/method_' num2str(p) '_' num2str(i) '.txt'],'r');
%                 for n=1:752
%                     data_l(n,:)=fscanf(fid, fmt, Inf)';
%                 end;
                %fclose(fid);
                %D=delimread([path '/' A_dir(k).name '/method_' num2str(p) '_' num2str(i) '.txt'],',','num');
                %data=D.num;
                %data=data_l;
                data=data';
                datap=[data(11,:) ; data(3,:) ; data(27,:) ; data(23,:) ; data(13,:) ; data(25,:) ; data(15,:) ; data(9,:) ; data(7,:) ; data(21,:) ; data(5,:) ; data(30,:) ; data(19,:) ; data(17,:)  ; data(1,:) ; data(2,:) ; data(4,:) ; data(20,:) ; data(22,:) ; data(8,:) ; data(18,:) ; data(29,:) ; data(10,:) ; data(16,:) ; data(26,:) ; data(6,:) ; data(24,:) ; data(12,:) ; data(28,:) ; data(14,:)];
                %data_innv(k-3,p,i,:,:)=datap';
                data_q=rescale(datap,-1,1);
                datap=datap';
                 if n==13
                     n=1;
                 end;
                 if p>=1 && p<=12
                        data_innv(k-3,1,n,i,1)=mean(mean(data_q));
                        pos1=min(find(t>=0));
                        pos2=max(find(t<=500));
                        data_innv(k-3,1,n,i,2)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=250));
                        pos2=max(find(t<=750));
                        data_innv(k-3,1,n,i,3)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=500));
                        pos2=max(find(t<=1000));
                        data_innv(k-3,1,n,i,4)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=750));
                        pos2=max(find(t<=1250));
                        data_innv(k-3,1,n,i,5)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=1000));
                        pos2=max(find(t<=1500));
                        data_innv(k-3,1,n,i,6)=mean(mean(data_q(:,pos1:pos2)));
                        n=n+1;
                       data_happy_trials(i,:,:)=datap+squeeze(data_happy_trials(i,:,:));
                 elseif p>=13 && p<=13+12
                        data_innv(k-3,2,n,i,1)=mean(mean(data_q));
                        pos1=min(find(t>=0));
                        pos2=max(find(t<=500));
                        data_innv(k-3,2,n,i,2)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=250));
                        pos2=max(find(t<=750));
                        data_innv(k-3,2,n,i,3)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=500));
                        pos2=max(find(t<=1000));
                        data_innv(k-3,2,n,i,4)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=750));
                        pos2=max(find(t<=1250));
                        data_innv(k-3,2,n,i,5)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=1000));
                        pos2=max(find(t<=1500));
                        data_innv(k-3,2,n,i,6)=mean(mean(data_q(:,pos1:pos2)));
                        n=n+1;
                       data_sad_trials(i,:,:)=datap+squeeze(data_sad_trials(i,:,:));
                 elseif p>=13+12+1 && p<=13+12+12+1
                        data_innv(k-3,3,n,i,1)=mean(mean(data_q));
                        pos1=min(find(t>=0));
                        pos2=max(find(t<=500));
                        data_innv(k-3,3,n,i,2)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=250));
                        pos2=max(find(t<=750));
                        data_innv(k-3,3,n,i,3)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=500));
                        pos2=max(find(t<=1000));
                        data_innv(k-3,3,n,i,4)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=750));
                        pos2=max(find(t<=1250));
                        data_innv(k-3,3,n,i,5)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=1000));
                        pos2=max(find(t<=1500));
                        data_innv(k-3,3,n,i,6)=mean(mean(data_q(:,pos1:pos2)));
                        n=n+1;
                        data_angry_trials(i,:,:)=datap+squeeze(data_angry_trials(i,:,:));
                 else
                        data_innv(k-3,4,n,i,1)=mean(mean(data_q));
                        pos1=min(find(t>=0));
                        pos2=max(find(t<=500));
                        data_innv(k-3,4,n,i,2)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=250));
                        pos2=max(find(t<=750));
                        data_innv(k-3,4,n,i,3)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=500));
                        pos2=max(find(t<=1000));
                        data_innv(k-3,4,n,i,4)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=750));
                        pos2=max(find(t<=1250));
                        data_innv(k-3,4,n,i,5)=mean(mean(data_q(:,pos1:pos2)));
                        pos1=min(find(t>=1000));
                        pos2=max(find(t<=1500));
                        data_innv(k-3,4,n,i,6)=mean(mean(data_q(:,pos1:pos2)));
                        n=n+1;
                        data_fear_trials(i,:,:)=datap+squeeze(data_fear_trials(i,:,:));
                end;
        end;
        for i=1:13
            data_happy_trials(i,:,:)=data_happy_trials(i,:,:)./12;
            data_sad_trials(i,:,:)=data_sad_trials(i,:,:)./12;
            data_angry_trials(i,:,:)=data_angry_trials(i,:,:)./12;
            data_fear_trials(i,:,:)=data_fear_trials(i,:,:)./12;
        end;
        data_average_trials=(data_happy_trials+data_sad_trials+data_angry_trials+data_fear_trials)./4;
    end;
    %clear data_innv
    data_average=data_average_trials+data_average;
    data_happy=data_happy_trials+data_happy;
    data_sad=data_sad_trials+data_sad;
    data_angry=data_angry_trials+data_angry;
    data_fear=data_fear_trials+data_fear;
    %n=n+1;
    save('data_inv_mat_ASD_all_whiten3.mat','data_average','data_happy','data_sad','data_angry','data_fear','data_innv');
    save('data_inv_mat_ASD_stats_whiten3.mat','data_innv');
end;
data_average=data_average./n;
data_happy=data_happy./n;
data_sad=data_sad./n;
data_angry=data_angry./n;
data_fear=data_fear./n;
save('data_inv_mat_ASD_all_whiten3.mat','data_average','data_happy','data_sad','data_angry','data_fear','data_innv');
save('data_inv_mat_ASD_stats_whiten3.mat','data_innv');
end;
if sel==0
    load('data_inv_mat_ASD_all.mat');
end;
for m=1:5 % classes including average
    figure(m)
    if m==1
        dataM=data_average;
    elseif m==2
        dataM=data_happy;
    elseif m==3
        dataM=data_sad;
    elseif m==4
        dataM=data_angry;
    else
        dataM=data_fear;
    end;
    for i=1:13 %% methods
        subplot(4,4,i);
        imagesc(rescale(squeeze(dataM(i,:,:)),-1,1))
        ax=gca;
        set(ax,'XTickLabel',{'250','500','750','1000','1250'});
        set(ax,'YTickLabel',{'F7','F3','FT9','FC5','T7','CP5','P7','O1','P3','CP1','C3','Cz','FC1','Fz', 'Fp1','Fp2','F4','FC2','CP2','P4','Pz','Oz','O2','P8','CP6','C4','FC6','F8','FT10','T8'})
        colormap(jet)
        title(methods{i})
        colorbar
    end;
end;
A=1;
