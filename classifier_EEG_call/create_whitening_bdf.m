function create_whitening_bdf(time_ranges)
t=linspace(-200,1550,875);
pos1=max(find(t<=time_ranges(1)));
pos2=min(find(t>=time_ranges(2)));
size_d=(pos2-pos1)+1;
DATA=load('bdf_trials_emotion_DANVA_Lerner_reDO_500.mat');
for k=1:35
    XDATA=prdataset(double([reshape(DATA.EEGk{k,1}.data(1:32,pos1:pos2,:),32*size_d,12)' ; reshape(DATA.EEGk{k,2}.data(1:32,pos1:pos2,:),32*size_d,12)' ; reshape(DATA.EEGk{k,3}.data(1:32,pos1:pos2,:),32*size_d,12)' ; reshape(DATA.EEGk{k,4}.data(1:32,pos1:pos2,:),32*size_d,12)']),[ones([1 12]) 2*ones([1 12]) 3*ones([1 12]) 4*ones([1 12])]');
     v_pos=[1:1:48];
     for l_pos=1:size(XDATA,1)
            Xtrain{l_pos}=XDATA(find(v_pos~=l_pos),:);
            Xtest{l_pos}=XDATA(find(v_pos==l_pos),:);
             %% PCA/ZCA whitening before sending it to Tensorflow
             %if sel_conv_read==1 && l_pos==1 && ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet\' subject_code{vector_pos(k)}],'dir')
             if l_pos==1
                    for p_c=1:size(Xtrain{l_pos}.data,1)
                        avg=mean(reshape(Xtrain{l_pos}.data(p_c,:),size_d,32),1);     % Compute the mean pixel intensity value separately for each patch. 
                        X=reshape(Xtrain{l_pos}.data(p_c,:),size_d,32)-repmat(avg,size(reshape(Xtrain{l_pos}.data(p_c,:),size_d,32),1), 1);
                        sigma_tr = X*X'/size(X,2);
                        [U_tr,S_tr,V_tr]=svd(sigma_tr);
                        xRot_tr=U_tr'*X;          
                        xTilde_tr=U_tr(:,1:10)'*X;
                        xPCAwhite_tr=diag(1./sqrt(diag(S_tr)+0.1))*U_tr'*X;
                        xZCAwhite_tr = U_tr * diag(1./sqrt(diag(S_tr) + 0.1))*U_tr'*X;
                        Xtrain_t{l_pos}(p_c,:)=reshape(xZCAwhite_tr,1,32*size_d);
                    end;
                    %xZCAwhite_tr
                    for q_c=1:size(Xtest{l_pos}.data,1)
                        avg=mean(reshape(Xtest{l_pos}.data(q_c,:),size_d,32),1);     % Compute the mean pixel intensity value separately for each patch. 
                        Xt=reshape(Xtest{l_pos}.data(q_c,:),size_d,32)-repmat(avg,size(reshape(Xtest{l_pos}.data(q_c,:),size_d,32),1), 1);
                        sigma_t = Xt*Xt'/size(Xt,2);
                        [U_t,S_t,V_tr]=svd(sigma_t);
                        xRot_t=U_t'*Xt;          
                        xTilde_t=U_t(:,1:10)'*Xt;
                        xPCAwhite_t=diag(1./sqrt(diag(S_t)+0.1))*U_t'*Xt;
                        xZCAwhite_t = U_t * diag(1./sqrt(diag(S_t) + 0.1))*U_t'*Xt;
                        Xtest_t{l_pos}(q_c,:)=reshape(xZCAwhite_t,1,32*size_d);
                    end;
                   %Ximag(ntd,:,:)=[[abs(Xtest_t{l_pos})] ; [abs(Xtrain_t{l_pos})]];
                   %ntd=ntd+1;
                   %% process the whitening images per subject
                    %% add the condition for folder existance
                   if ~exist(['C:\Users\Marcela Murillo\Documents\Code_Matlab\lerner_data\subject_' num2str(k)],'dir')
                        mkdir(['C:\Users\Marcela Murillo\Documents\Code_Matlab\lerner_data\subject_' num2str(k)]);
                        csvwrite(['C:\Users\Marcela Murillo\Documents\Code_Matlab\lerner_data\subject_' num2str(k) '\data_sub_' num2str(k) '.csv'],[[abs(Xtest_t{l_pos})  getlabels(Xtest{l_pos})] ; [abs(Xtrain_t{l_pos})  getlabels(Xtrain{l_pos})]]);
                    %if ~exist(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}],'dir')
                    %    mkdir(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)}]);
                    %    csvwrite(['C:\Users\jmayortorres\Documents\EEG_main_folder\CSV_files_training_ConvNet_Whole_Trial\' subject_code{vector_pos(k)} '\data_sub_' subject_code{vector_pos(k)} '.csv'],[[abs(Xtest_t{l_pos})  getlabels(Xtest{l_pos})] ; [abs(Xtrain_t{l_pos})  getlabels(Xtrain{l_pos})]]); 
                  % end
               end;
        end;
   end;
end;