function read_ALL_subjects_folders(path)
%% This code reads and apply the denoising and artifact rejection for all the data presented in the DANVA and ToMs dataset from StonyBrook University - Social Competence and trialeatment Lab (SCTL)
%% path: path where we can see the DANVA folders with corresponding code such as 1730006 or 1730004 etc..The .egg and the .vmrk and the .vhdr files are contained in this file. Change the '\' for '/' if you will replicate this code in Linux
%% divide the folders inside the DANVA folders and A_dir(k).name
A_dir=dir(path)
np=1;
%% set this depending when the DANVA path start organizing the .mat files per folder
for k=3:length(A_dir)
    for trial=1:48 %% exclude the corresponding trial from the test set to establish the LOTO cross-validation per subject
        if exist([path '\' A_dir(k).name])==7 && length(A_dir(k).name)>=3 && ~exist([[path '/' A_dir(k).name] '\DANVA_res.mat'],'file') %&& isempty(strialt.EEG_tom_correct)%&& ~exist([[path '/' A_dir(k).name] '\DANVA_res.mat'],'file') %&& length(A_dir(k).name)<=6
             fclose all;
             [path '\' A_dir(k).name]
             k
             %% here we need to be careful to separate the test trialial from the rest before doing the EEGlab preprocessing!! 
             EEG_val_happy=BV_EEGlab_comp_all([path '\' A_dir(k).name],10,0,1e-5,40,32,{'S  1','S  2','S 11','S 12'},[-200 1550],[750 1200],1,trial);
             EEG_val_sad=BV_EEGlab_comp_all([path '\' A_dir(k).name],10,0,1e-5,40,32,{'S  3','S  4','S 13','S 14'},[-200 1550],[750 1200],1,trial);
             EEG_val_angry=BV_EEGlab_comp_all([path '\' A_dir(k).name],10,0,1e-5,40,32,{'S  5','S  6','S 15','S 16'},[-200 1550],[750 1200],1,trial);
             EEG_val_fear=BV_EEGlab_comp_all([path '\' A_dir(k).name],10,0,1e-5,40,32,{'S  7','S  8','S 17','S 18'},[-200 1550],[750 1200],1,trial);
             %% remember to save the DANVA results file with the suffix of test trial to take it into account in the ZCA code
             save([[path '/' A_dir(k).name] '\DANVA_res_' num2str(trial) '.mat'],'EEG_val_happy','EEG_val_sad','EEG_val_angry','EEG_val_fear');
             fclose all;
             EEG_tom_correct=BV_EEGlab_comp_all([path '\' A_dir(k).name],10,1,1e-5,40,32,{'S202'},[-200 1550],[750 1200],1,trial); 
             EEG_tom_no_correct=BV_EEGlab_comp_all([path '\' A_dir(k).name],10,1,1e-5,40,32,{'S200'},[-200 1550],[750 1200],1,trial);
             save([[path '/' A_dir(k).name] '\TOM_res_' num2str(trial) '.mat'],'EEG_tom_correct','EEG_tom_no_correct');
             fclose all;
             A_name{np}=A_dir(k).name;
             np=np+1;
        end;
    end;
end;
