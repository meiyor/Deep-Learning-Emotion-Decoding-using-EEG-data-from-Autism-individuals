function read_perf_convnets_subjects(dat_name,path)
A_dir=dir(path);
Cn=zeros([4 4]);
CCn=zeros([4 4]);
close all
load('dat_e.mat','dat_emo');
%% use this trials to measure metrics only for intensity level 2 %% only for intensity level
index_trials=[0 1 0 1 1 0 1 1 0 1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 1 1 0 0 0 0 1 1 1 1 0 1 1 0 1 0 0 0 1 0 1 0];
index_trials2=[index_trials(find(dat_emo==1)') index_trials(find(dat_emo==2)') index_trials(find(dat_emo==3)') index_trials(find(dat_emo==4)')];
for k=4:length(A_dir)
    C{k-2}=zeros([4 4]);
    Cc{k-2}=zeros([4 4]);
    str_p{k}=[path '/' A_dir(k).name];
    str_s=strsplit(A_dir(k).name,'_')
    file_path=str_p{k}
    subject_code{k-2}=str_s{2};
    wf=fopen(file_path,'r');
    n=1;
    nc=1;
    while exist(file_path,'file') && ~feof(wf) 
     sampl_2=0;   
     str_header{n,k-2}=fgets(wf); 
     if ~contains(str_header{n,k-2},'trial')
         str_header{n,k-2}=fgets(wf);
     else
         vals_pos=find(index_trials2==1);
         %% check the trial index
         if length(str_header{n,k-2})==8
             pos_ind=str2num(str_header{n,k-2}(end-2));
             if any(vals_pos==pos_ind)
                sampl_2=1;
             end; 
         elseif length(str_header{n,k-2})==9
             pos_ind=str2num(str_header{n,k-2}(end-3:end-2));
             if any(vals_pos==pos_ind)
                sampl_2=1;
             end;
         elseif length(str_header{n,k-2})>=9  
             posj=find(str_header{n,k-2}=='l')
             pos_ind=str2num(str_header{n,k-2}(posj+1:posj+2));
             if any(vals_pos==pos_ind)
                sampl_2=1;
             end;
         end;
     end;
     pos_i=find(str_header{n,k-2}=='l');
     if ~isempty(str2num(str_header{n,k-2}(pos_i+2))) %|| (str2num(str_header{n,k-2}(pos_i+2))<=9 && str2num(str_header{n,k-2}(pos_i+2))>=0)
        num_c=str2num(str_header{n,k-2}(pos_i+1:pos_i+2));
     else
        num_c=str2num(str_header{n,k-2}(pos_i+1)); 
     end;
     if num_c<=12
        class_del(n,k-2)=1;
     elseif num_c>12 && num_c<=24
        class_del(n,k-2)=2; 
     elseif num_c>24 && num_c<=36
        class_del(n,k-2)=3;  
     else
        class_del(n,k-2)=4;   
     end;
     str_data=fgets(wf);
     str_data_unit=strsplit(str_data,',');
     if ~isempty(str_data_unit{1})
        data_(n,k-2)=str2num(str_data_unit{1}(end-2:end));
        if sampl_2==1
            datac_(nc,k-2)=str2num(str_data_unit{1}(end-2:end));
        end;
        if length(str_data_unit)>=3
            dat1=strsplit(str_data_unit{3},'=')
        else
             str_data=fgets(wf);
             str_data_unit=strsplit(str_data,',');
             if ~isempty(str_data_unit{1})
                dat1=strsplit(str_data_unit{1},'=');
             else
                dat1=strsplit(str_data_unit{2},'='); 
             end;
        end;
        dat2=strsplit(dat1{2},'[');
        dat3=strsplit(dat2{2},']');
        prob=strsplit(dat3{1},' ');
        probv=[str2num(prob{2}) str2num(prob{3}) str2num(prob{4}) str2num(prob{5})];
        probm=sort(probv);
        class(n,k-2)=find(probv==probm(end))
        if sampl_2==1
             classc(nc,k-2)=find(probv==probm(end))
             classc_del(nc,k-2)=class_del(n,k-2);
             Cc{k-2}(classc(nc,k-2),class_del(n,k-2))=Cc{k-2}(classc(nc,k-2),class_del(n,k-2))+1;
         end;
     else
         data_(n,k-2)=str2num(str_data_unit{2}(end-2:end));
         if length(str_data_unit)>=3
            dat1=strsplit(str_data_unit{3},'=')
         else
             str_data=fgets(wf);
             str_data_unit=strsplit(str_data,',');
             if ~isempty(str_data_unit{1})
                dat1=strsplit(str_data_unit{1},'=');
             else
                dat1=strsplit(str_data_unit{2},'='); 
             end;
         end;
         dat2=strsplit(dat1{2},'[');
         dat3=strsplit(dat2{2},']');
         prob=strsplit(dat3{1},' ');
         probv=[str2num(prob{2}) str2num(prob{3}) str2num(prob{4}) str2num(prob{5})];
         probm=sort(probv);
         class(n,k-2)=find(probv==probm(end))
         if sampl_2==1
             classc(nc,k-2)=find(probv==probm(end))
             classc_del(nc,k-2)=class_del(n,k-2);
             Cc{k-2}(classc(nc,k-2),class_del(n,k-2))=Cc{k-2}(classc(nc,k-2),class_del(n,k-2))+1;
             nc=nc+1;
         end;
     end;
     C{k-2}(class(n,k-2),class_del(n,k-2))=C{k-2}(class(n,k-2),class_del(n,k-2))+1;
     n=n+1;
     if sampl_2==1
        nc=nc+1;
     end;
     if contains(str_header{n-1,k-2},'trial9')
         break;
     end;
    end;
  Cn=C{k-2}+Cn;
  CCn=Cc{k-2}+CCn;
  [pr(k-2),re(k-2)]=prrecalc(C{k-2},4);
  [prc(k-2),rec(k-2)]=prrecalc(Cc{k-2},4);
  fclose all
end;
acc=sum(data_(1:48,:))./48;
error1=sum(1-data_(1:48,:));
error1_s=sum(sum(1-data_(1:48,:)));
error1_m=mean(sum(1-data_(1:48,:)));
acc_c=sum(datac_(1:24,:))./24;
save([dat_name '.mat'],'acc_c','prc','rec','acc','pr','re');
figure;
plot_confu_comp_v(Cn,{'Happy','Sad','Angry','Fear'},12*size(data_,2),4)
figure;
plot_confu_comp_v(CCn,{'Happy','Sad','Angry','Fear'},6*(size(datac_,2)),4)
save([dat_name '.mat'],'acc_c','prc','rec','acc','pr','re');
mean(pr)
mean(re)
mean(prc)
mean(rec)
fclose all
