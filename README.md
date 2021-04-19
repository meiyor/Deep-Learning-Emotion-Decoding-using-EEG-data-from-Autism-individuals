# Deep Learning Emotion decoding using EEG data from Autism individuals

This repository includes the python and matlab codes using for processing EEG 2D images on
a customized Convolutional Neural Network (CNN) to decode emotion visual stimuli on individuals with and without
Autism Spectrum Disorder (ASD).

If you would like to use this repository to replicate our experiments with this data or use your our own data, please cite the following paper, more details about this code and implementation are described there as well:

**Mayor Torres, J.M. ¥**, **Clarkson, T.¥**, Hauschild, K.M., Luhmann, C.C., Lerner, M.D., Riccardi,  G., [**Facial emotions are accurately encoded in the brains of those with autism: A deep learning approach.**](https://www.sciencedirect.com/science/article/pii/S2451902221001075?via%3Dihub) Biological Psychiatry: Cognitive Neuroscience and Neuroimaging,(2021).

### Requirements
- Tensorflow >= v1.20
- sklearn
- subprocess
- numpy
- csv
- Matlab > R2018b

For the python code we provide:

__1.__ A baseline code to evaluate a Leave-One-Trial-Out cross-validation from two csv files. One including all the trials for train with their corresponding labels and other with the test features of the single trial you want to evaluate. The test and train datafile should have an identifier to be paired by the for loop used for the cross validation. The code to run the baseline classifiier is located on the folder **classifier_EEG_call**.

![Pipeline for EEG Emotion Decoding](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/pipeline_2_using_latex.jpeg)

  To run the classifier pipeline simply download the .py files on the folder **classifier_EEG_call** and execute the following command on your bash prompt:
  
```python 
   python LOTO_lauch_emotions_test.py "data_path_file_including_train_test_files"
```
Please be sure your .csv files has a flattened time-points x channels EEG image after you remove artifacts and noise from the signal. Using the ADJUST EEGlab pipeline preferrably (https://sites.google.com/a/unitn.it/marcobuiatti/home/software/adjust).

The final results will be produced in a txt file as we provide in the **examples** folder. Some metrics obtained from a sample of 88 ADOS-2 diagnosed participants 48 controls, and 40 ASD are the following:

| Metrics/Groups 	| FER    	|        	|        	|       	| CNN   	|       	|       	|       	|
|----------------	|--------	|--------	|--------	|-------	|-------	|-------	|-------	|-------	|
|                	| Acc    	| Pre    	| Re     	| F1    	| Acc   	| Pre   	| Re    	| F1    	|
| TD             	| 0.813  	| 0.808  	| 0.802  	| 0.807 	| 0.860 	| 0.864 	| 0.860 	| 0.862 	|
| ASD*           	| 0.776  	| 0.774  	| 0.768  	| 0.771 	| 0.934 	| 0.935 	| 0.933 	| 0.934 	|

Face Emotion Recognition (FER) task performance is denoted as the human performance obtained when labeling the same stimuli presented to obtain the EEG activity.

__2.__ A code for using the package the iNNvestigate package (https://github.com/albermax/innvestigate) Saliency Maps and unify them from the LOTO crossvalidation mentioned in the first item. Code is located in the folder **iNNvestigate_evaluation**

To run the investigate evaluation simply download the .py files on the folder **iNNvestigate_evaluation** and execute the following command on your bash prompt:
  
```python 
   python LOTO_lauch_emotions_test_innvestigate.py "data_path_file_including_train_test_files" num_method
```

The value __num_method__ is defined based on the order iNNvestigate package process saliency maps. For our specific case the number concordance is: 

'Original Image'-> 0 'Gradient' -> 1 'SmoothGrad'-> 2 'DeconvNet' -> 3 'GuidedBackprop' -> 4 'PatterNet' -> 5 'PatternAttribution' -> 6 'DeepTaylor' -> 7 'Input * Gradient' -> 8 'Integrated Gradients' -> 9 'LRP-epsilon' -> 10 'LRP-Z' -> 11 'LRP-APresetflat' -> 12 'LRP-BPresetflat' -> 13

Some average masks calculated from the iNNvestigate methods are included in **examples** folder. Each contributor or user is free to play with them.

#### An example from saliency maps obtained from LRP-B preset are shown below):
#### significant differences are observed on 750-1250 ms relative to the onset between the relevance of Controls and ASD groups! 

![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_TD_def.jpeg)
![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_ASD_def.jpeg)
![alt text](https://github.com/meiyor/Deep-Learning-Emotion-Decoding-using-EEG-data-from-Autism-individuals/blob/master/LRP-BPresetflat_Average_diff_def.jpeg)

For the Matlab code we provide the repository for reading the resulting output performance files for the CNN baseline classifier **Reading_CNN_performances**, and for the iNNvestigate methods using the same command call due to the output file is composed of the same syntax.

To run a performance checking first download the files on **Reading_CNN_performances** folder and run the following command on your Matlab prompt sign having the results .csv files from the folder **examples**.

```matlab 
   read_perf_convnets_subjects('suffix_file','performance_data_path')
```
Take into account for the results attached in the **examples** folder use the __test_t_performance_test__ suffix and the code will read the csvs automatically.
