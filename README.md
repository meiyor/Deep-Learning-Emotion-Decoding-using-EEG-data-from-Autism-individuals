# Deep Learning Emotion decoding using EEG data from Autism individuals

These codes includes the python and matlab codes using for processing EEG 2D images on
a customized Deep ConvNet to decode emotion visual stimuli on individuals with and without
Autism Spectrum Disorder (ASD).

### Requirements
- Tensorflow >= v1.20
- sklearn
- subprocess
- numpy
- csv
- Matlab > R2018b

For the python code we provide:

__1.__ A baseline code to evaluate a Leave-One-Trial-Out cross-validation from two csv files. One including all the trials for train with their corresponding labels and other with the test features of the single trial you want to evaluate. The test and train datafile should have an identifier to be paired by the for loop used for the cross validation. The code to run the baseline classifiier is located on the folder **classifier_EEG_call**

![Pipeline for EEG Emotion Decoding](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/pipeline_2_using_latex.jpeg)

  To run the classifier pipeline simply download the .py files on the folder **classifier_EEG_call** and execute the following command on your bash prompt:
  
```python 
   python LOTO_lauch_emotions_test.py "data_path_file_including_train_test_files"
```
Please be sure your .csv files has a flattened time-points x channels EEG image after you remove artifacts and noise from the signal. Using the ADJUST EEGlab pipeline preferrably (https://sites.google.com/a/unitn.it/marcobuiatti/home/software/adjust).

The final results will be produced in a txt file as we provide in the **examples** folder.

__2.__ A code for using the package the iNNvestigate package (https://github.com/albermax/innvestigate) Saliency Maps and unify them from the LOTO crossvalidation mentioned in the first item. Code is located in the folder **iNNvestigate_evaluation**

For the Matlab code we provide the repository for reading the resulting output performance files for the CNN baseline classifier **Reading_CNN_performances**, and for the iNNvestigate methods **Reading_iNNvestigate_performances**

![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_TD_def.jpeg)
![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_ASD_def.jpeg)
![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_diff_def.jpeg)
