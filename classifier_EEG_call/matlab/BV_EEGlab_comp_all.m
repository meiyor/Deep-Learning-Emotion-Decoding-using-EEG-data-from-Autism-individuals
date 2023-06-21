function [EEGd]=BV_EEGlab_comp_all(path,k_ind,sel_ToM_Danva,l_rate_ica,steps_ica,num_channels,stim_p,p_time_range,p_time_fix,sel_exclude_EOG)
%% USE THIS CODE ONLY FOR CORRECTED MARKED FILES, IF THIS ARE NOT CORRECTLY MARKED THIS CODE SHOULD AVOID THAT FILES FOR DOING MARKER RECOVERY YOU SHOULD USE BV_EEGlab_QK.m!!
%% path: Single subject BrainVision absolute path. This folder may contain .eeg, .vhdr, and .mrk files
%% k_ind: moving average filter order using fma EEGlab function
%% before run this function add the bvaio path into the Matlab path
%% sel_ToM_Danva: select to evaluate if the markers are missing in the DANVA marker file 2 -> missing markers, 0 -> complete DANVA_trial,  1 -> complete ToM_trial, .
%% l_rate_ica: learning rate for ICA decomposition.
%% steps_ica: ICA number of iterations in the decomposition process.
%% num_channels: number of channels from the BrainProducts Acticap
%% stim_p: stimuli cell array the length of the stimuli set should be less or equal to the number of .eeg files in the path_k folder
%% p_time_range: epoching time range in milliseconds, baseline will be removed from p_time_range(1) to the stim onset.
%% p_time_fix: averaging time range for topoplot in milliseconds.
%% sel_exclude_EOG: this selector allow you to exclude the EOG channels from the beginning of the analysis 0-> including,1 -> not including.
close all;

%addpath(genpath('C:\Users\psychuser\Documents\bvaio1.57')); %% use this
%line if you have the absolute path for the bvaio.57 plugin

A_dir=dir(path); %% reading the directory file structure from the server (NO name modifications if possible!!)
header_e={'type','latency'};
m_count=1;
n_cells=0; %% assuming that the number of n_cells in the stimuli set is at least one
for p=1:length(stim_p)
    if (iscell(stim_p{p}))
        n_cells=n_cells+1;
    end;
end;
if n_cells==0 %% assuming that this is not any cell inside the global cell array
    n_cells=1;
end;
for i=1:length(A_dir)
    indicator_proc=0;
    F1=fopen('Standard-10-20-Cap81.locs','r');
    F2=fopen('pos_10_20_2.loc','w');
    FP=fopen('event_DANVA.txt','w');
    FQ=fopen('event_DANVA_alt.txt','w');
    if (exist([path '/' A_dir(i).name])~=7 && length(A_dir(i).name)>=4 && all(A_dir(i).name(end-4:end)=='.vhdr')) %% count the number of files in the folder that contains a valid BV folder
        [EEG{i},com]=pop_loadbv(path,A_dir(i).name);
        if ~isfield(EEG{i},'setname')
            [EEG{i},com]=pop_loadbv(path,A_dir(i).name);
        end;
         Str=strsplit(path,'\');
        if sel_ToM_Danva==0 %% DANVA
            %% set the bandpass filter to 0.1 - 30 Hz ranges
            band_filter=[0.1 30];
            %if all(A_dir(i).name(1:5)=='DANVA')
               indicator_proc=1;
            %end;
        elseif sel_ToM_Danva==1 %% ToM trial
            %% set the bandpass filter to 0.1 - 30 Hz ranges
            band_filter=[0.1 30];
            if all(A_dir(i).name(1:3)=='ToM') || contains(A_dir(i).name,'T')
                indicator_proc=1;
            end;
        elseif sel_ToM_Danva==2 %% wet_run Aliengonogo
            %% set the bandpass filter to 0.1 - 30 Hz ranges
            band_filter=[0.1 30];
            if all(A_dir(i).name(1:3)=='wet')
                indicator_proc=1;
            end;
        end;
       %%if (length(EEG{i}.event)>=3 || (length(EEG{i}.event)<=3 && sel_ToM_Danva==1))
       if (length(EEG{i}.event)>=3 && indicator_proc==1)
            n2=1;
            while (~feof(F1))
                    for (kk=1:4)
                            Data_locs{n2,kk}=fscanf(F1,'%s\t',1);
                    end;
            n2=n2+1;
            end;
            fclose(F1);
            check_lat=return_latencies(EEG{i},{'S 22'});
            if isempty(check_lat)
                EEGd=[];
                %return;
            else
                fk=fopen('markers_nomarkers_new.txt','w');
                bias=abs(randperm(48)/500)*20;
                for kn=1:48
                    fprintf(fk,'%s %s\n','marker_n',num2str(check_lat(1)+10+bias(kn)));
                end;
                fclose(fk);
                EEG{i}=pop_importevent(EEG{i},'event','markers_nomarkers_new.txt','fields',header_e);
            end;
            for h=1:num_channels-1
                p_index=find(ismember({Data_locs{:,4}},EEG{i}.chanlocs(h).labels)==1); %% use it if the version is older than R2015a
                %p_index=find(contains({Data_locs{:,4}},EEG{i}.chanlocs(h).labels)); %% use it if the version is newer than R2015a
                if (length(p_index)>=1)
                    for m=1:length(p_index)
                        if (length(Data_locs{p_index(m),4})==length(EEG{i}.chanlocs(h).labels))
                            p_index=p_index(m);
                            break;
                        end;
                    end;
                end;
                %num2str(str2num(DD{h,2})*180/pi)
                fprintf(F2,'%s %s %s %s\n',num2str(h),Data_locs{p_index,2},Data_locs{p_index,3},EEG{i}.chanlocs(h).labels);
            end;
        fprintf(F2,'%s %s %s %s\n',num2str(num_channels),num2str(20),num2str(0.61),EEG{i}.chanlocs(num_channels).labels);
        fprintf(F2,'%s %s %s %s\n',num2str(num_channels+2),num2str(-20),num2str(0.61),EEG{i}.chanlocs(num_channels+1).labels);
        fclose(F2);
        t_size=size(EEG{i}.data,2);
        EEG{i}=pop_editset(EEG{i},'setname','Data1','chanlocs','pos_10_20_2.loc','icaweights',[]);
        EEG{i}.chanlocs=readlocs('pos_10_20_2.loc');
        if sel_exclude_EOG==1 %% select this for new TP9_10 re-reference
            EEG{i}.data=EEG{i}.data(1:31,:);
            EEG{i}.chanlocs=EEG{i}.chanlocs(1:31);
            EEG{i}.nbchan= EEG{i}.nbchan-2;
            re_locs=readlocs('Standard-10-20-Cap81.locs');
            EEG{i}=pop_reref(EEG{i},[27 28],'refloc',re_locs(39)); %%re-reference
        end;
        EEG{i} = pop_eegfiltnew(EEG{i},band_filter(1),band_filter(2),16500);
        
        %rmpath(genpath('/Users/juan_m_mayor_trento/Documents/HMOproj1_ver/HMOprojectRTT/EEG/eeglab13_4_4b'))
        %rmpath(genpath('/Users/juan_m_mayor_trento/Documents/HMOproj1_ver/HMOprojectRTT/EEG/eeglab13_4_4b/fieldtrip-20160917'))
        for k=1:num_channels-2
                t_data=conv((1/k_ind)*ones([1 k_ind]),EEG{i}.data(k,:));
                t_data=detrend(t_data);
                EEG{i}.data(k,:)=t_data(k_ind/2:t_size+(k_ind/2-1));
        end;
        if (str2num(Str{end}(end-1:end))>=1)
           EEG{i}=clean_rawdata(EEG{i},5,[0.25 0.75],0.85,-1,-1,-1);
        end;
         %EEGd{m_count}.data(32:33,:,:)=zeros([2 size(EEG{i}.data,2) size(EEG{i}.data,3)]);
         if l_rate_ica==0 && steps_ica==0
            [EEG{i}.icaweights,EEG{i}.icasphere]=runica(EEG{i}.data(:,:),'sphering','on');
         else
            [EEG{i}.icaweights,EEG{i}.icasphere]=runica(EEG{i}.data(:,:),'sphering','on','lrate',l_rate_ica,'maxsteps',steps_ica); 
         end;
         EEG{i}.icawinv=inv(EEG{i}.icaweights*EEG{i}.icasphere);
        %% add Cz position for interpolation
        % EEG{i}.data(end,:)=(EEG{i}.data(19,:)+EEG{i}.data(20,:)+EEG{i}.data(21,:)+EEG{i}.data(22,:))/4%%linear interpolation around Cz
        %% Do the epoching!! 
         EEGtemp=EEG{i};
         if length(stim_p)==1
            EEG{i}=pop_epoch(EEG{i},stim_p,[p_time_range(1) p_time_range(2)]./1000); %% it is divided by 1000 because for pop 
         else
           if n_cells==1  
             EEG{i}=pop_epoch(EEG{i},stim_p,[p_time_range(1) p_time_range(2)]./1000);
           else
             EEG{i}=pop_epoch(EEG{i},stim_p{m_count},[p_time_range(1) p_time_range(2)]./1000);
           end;
           if (isempty(EEG{i}))
               EEGd=[];
               return;
           end;
           %% validate events size in the case of DANVA-ToM data repositories
            if length(EEG{i}.event)<3 && length(EEGtemp.event)<3
                if (m_count>=length(stim_p))
                    EEG{i}=pop_epoch(EEG{i},{stim_p{m_count-1}},[p_time_range(1) p_time_range(2)]./1000);  
                else
                    EEG{i}=pop_epoch(EEG{i},{stim_p{m_count+1}},[p_time_range(1) p_time_range(2)]./1000);
                end;
            end;
         end;
         if isempty(EEG{i}) || isempty(EEG{i}.event)
             EEGd=[];
             break;
         end;
         
          %% Run ADJUST after the ICA decomposition    
         if (length(size(EEG{i}.data))<3)
             EEG{i}.data= repmat(EEG{i}.data(:,:),1,1,5);
             EEG{i}.trials=5;
         end;
         [art_channels]=ADJUST(EEG{i},'report.txt');
         if ~(any(art_channels==1) && (any(art_channels==2)) && length(EEG{i}.chanlocs(1).labels)==3 && length(EEG{i}.chanlocs(2).labels)==3 && all(EEG{i}.chanlocs(1).labels=='Fp1') && all(EEG{i}.chanlocs(2).labels=='Fp2'))
                EEG{i}=pop_subcomp(EEG{i},[art_channels 1 2]); %% force to remove the frontal channels subcomps if the are included in the list of artifact channels
         else
                EEG{i}=pop_subcomp(EEG{i},art_channels);
         end;
         EEG{i}=pop_rmbase(EEG{i},[p_time_range(1)  0]);
         t=linspace(p_time_range(1),p_time_range(2),size(EEG{i}.data,2));
         figure;
         if (num_channels-2<=size(EEG{i}.data,1))   
            plot(t,squeeze(mean(EEG{i}.data(1:num_channels-2,:,:),3))');
         else
            plot(t,squeeze(mean(EEG{i}.data(1:size(EEG{i}.data,1),:,:),3))'); 
         end;
         xlabel('Time [s]');
         ylabel('Amplitude uV');
         grid on;
         %EEG{i}.data=-1*EEG{i}.data; %% comment this line if you don't want to make a data inversion  
         EEGd{m_count}=EEG{i};
         EEGd{m_count}.data(32:33,:,:)=zeros([2 size(EEG{i}.data,2) size(EEG{i}.data,3)]);
         figure;
         pop_timtopo(EEGd{m_count},[p_time_range(1) p_time_range(2)-50],[-20],'ERP','plotchans',[1:num_channels-3],'electrodes','labels')
         figure;
         p1=max(find(t<=p_time_fix(1)));
         p2=min(find(t>=p_time_fix(2)));
         topoplot(mean(mean(EEGd{m_count}.data(:,p1:p2,:),3),2),EEGd{m_count}.chanlocs,'electrodes','on','electrodes','labels');
         save('EEG_1.mat','EEGd');
         %figure;
         %% itpc processing %% if this necessary ***
         %[erspp,itckp,powbase,timeskp,freqp]=newtimef(mean(EEGd{m_count}.data(:,:,:),1),size(EEGd{m_count}.data,2),[p_time_range(1) p_time_range(2)-50],EEGd{m_count}.srate,0,'itctype','phasecoher','winsize',200,'freqs',[1 20],'nfreqs',20,'padratio',8,'plotersp','off');
         %for (i=1:num_channels-2) %% moving around all the channels
         %  [ersp{i},itck{i},powbasep{i},timesk{i},freqs{i}]=newtimef(squeeze(EEGd{m_count}.data(i,:,:)),size(EEGd{m_count}.data,2),[p_time_range(1) p_time_range(2)-50],EEGd{m_count}.srate,0,'itctype','phasecoher','winsize',200,'nfreqs',20,'padratio',8,'freqs',[1 20],'plotersp','off','plotitc','off')
         %end;
         if m_count>=n_cells
             break;
         end;
         m_count=m_count+1;
         fclose all; %% close all the remaining malloc pointers generated by pop_bva function
       else
         clear EEG; %%if not this means that the DANVA file has the markers missing from .vmrk file
         EEGd=[];
         disp('No Markers in this file!');
       end;
    end;
end;
