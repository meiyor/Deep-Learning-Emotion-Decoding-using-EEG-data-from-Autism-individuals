# Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals

These codes includes the python and matlab codes using for processing EEG 2D images on
a customized Deep ConvNet to decode emotion visual stimuli on individuals with and without
Autism Spectrum Disorder (ASD).

For the python code we provide:

__1.__ A baseline code to evaluate a Leave-One-Trial-Out cross-validation from two csv files. One including all the trials for train with their corresponding labels and other with the test features of the single trial you want to evaluate. The test and train datafile should have an identifier to be paired by the for loop used for the cross validation. The code to run the baseline classifiier is located on the folder **classifier_EEG_call**

__2.__ A code for using the package the iNNvestigate package (https://github.com/albermax/innvestigate) Saliency Maps and unify them from the LOTO crossvalidation mentioned in the first item. Code is located in the folder **iNNvestigate_evaluation**


![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_TD.jpg)
![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_ASD.jpg)
![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_diff.jpg)
