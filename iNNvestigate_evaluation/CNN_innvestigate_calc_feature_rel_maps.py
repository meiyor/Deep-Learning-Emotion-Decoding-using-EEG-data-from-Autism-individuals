# coding: utf-8
## this code use keras-tensorflow backend for definying the CNN for EEG emotion recognition in the study - Mayor Torres, J.M. ¥, Clarkson, T.¥, Hauschild, K.M., Luhmann, C.C., Lerner, M.D., Riccardi, G., Facial emotions are accurately encoded in the brains of those with autism: A deep learning approach. Biological Psychiatry: Cognitive Neuroscience and Neuroimaging,(2021).
## written by: Juan Manuel Mayor-Torres
##

import imp
import numpy as np
import os

import keras
import keras.backend
import keras.models

keras.backend.tensorflow_backend._get_available_gpus()
import innvestigate
import innvestigate.utils as iutils

import csv
import sys
from numpy import genfromtxt

csvFile1 = str(sys.argv[1]) # training and test files taking per subject specifiy these files in the LOTO CV in an upper level - constructed in Matlab
csvFile2 = str(sys.argv[2])

training_data_file = genfromtxt(csvFile1, delimiter=',', skip_header=0) ## reading training file for each trial
test_data_file = genfromtxt(csvFile2, delimiter=',', skip_header=0) ## reading test file for each trial

data_shape = "There are " + repr(training_data_file.shape[0]) + " samples of vector length " + repr(training_data_file.shape[1])
num_rows = training_data_file.shape[0] # Number of data samples
num_cols = training_data_file.shape[1] # Length of Data Vector

num_rowst = 1 # Number of data samples
num_colst = test_data_file.shape[0] #test_data_file.shape[1] # Length of Data Vector

total_size_train=(num_cols-1)*num_rows
total_size_test=(num_colst-1)*num_rowst

data_train = np.arange(total_size_train)
data_train = data_train.reshape(num_rows, num_cols-1)

data_train = data_train.astype('float64')

label_train = np.arange(num_rows)
label_train = label_train.astype('int32')

data_test = np.arange(total_size_test)
data_test = data_test.reshape(num_rowst, num_colst-1) # 2D Matrix of data points
data_test = data_test.astype('float64')

label_test = np.arange(num_rowst)
label_test = label_test.astype('int32')

## convert the .csv file data in a numpy array for training and..
for i in range(training_data_file.shape[0]):
             label_train[i] = training_data_file[i][num_cols-1]
             for j in range(num_cols-1):
                 data_train[i][j] = training_data_file[i][j]

## for test respectively..
for i in range(1):
         #print(test_data_file.shape)
            label_test[i] = test_data_file[num_colst-1]
            for j in range(num_colst-1):
                data_test[i][j] = test_data_file[j]

data_train=data_train.astype('float32',casting='same_kind')
label_train=label_train.astype('int32',casting='same_kind')-1
data_test=data_test.astype('float32',casting='same_kind')
label_test=label_test.astype('int32',casting='same_kind')-1

##reshape with the shape given in Matlab
data_train=np.reshape(data_train,(47,752,30,1))
data_test=np.reshape(data_test,(1, 752,30, 1))
import warnings
warnings.simplefilter('ignore')

# Use utility libraries to focus on relevant iNNvestigate routines.
# change the utils and utils_mnist depending on the iNNvestigate version

mnistutils = imp.load_source("utils_mnist", "../utils_mnist.py")


#### Data loading in Keras

data=(data_train, label_train, data_test, label_test)

input_range = [0.1, 0.5] ##only for deep-taylor bounded 

num_classes = 4
label_to_class_name = ["happy","sad","angry","fear"]


# ## Model
# The next part trains and evaluates a CNN

# Create and train model
if keras.backend.image_data_format == "channels_first":
    input_shape = (1, 752, 30)
else:
    input_shape = (752, 30, 1)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (100, 10), activation="relu", padding="same", kernel_initializer=keras.initializers.glorot_uniform(seed=None),bias_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),input_shape=input_shape),
    keras.layers.MaxPooling2D((5, 2),strides=(2,2),padding="same"),
    keras.layers.Conv2D(64, (20, 5), activation="relu", padding="same", kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.RandomNormal(mean=0.0 , stddev=0.1, seed=None)),
    keras.layers.MaxPooling2D((2, 2),strides=(2,2),padding="same"),
    keras.layers.Conv2D(128, (10, 2), activation="relu", padding="same", kernel_initializer=keras.initializers.glorot_uniform(seed=None), bias_initializer=keras.initializers.RandomNormal(mean=0.0 , stddev=0.1, seed=None)),
    keras.layers.MaxPooling2D((2, 2),strides=(2,2),padding="same"),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(1024, activation="relu"),
    keras.layers.Dense(4, activation="softmax"),
])

##performance
scores = mnistutils.train_model(model, data, batch_size=4, epochs=30)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))


# ## Analysis

# Next, we will set up a list of analysis methods by preparing tuples containing the methods' string identifiers used by `innvestigate.analyzer.create_analyzer(...)`, some optional parameters, a post processing choice for visualizing the computed analysis and a title for the figure to render. Analyzers can be deactivated by simply commenting the corresponding lines, or added by creating a new tuple as below.
# 
# For a full list of methods refer to the dictionary `investigate.analyzer.analyzers`


def input_postprocessing(X):
    return X
    ## is not necessary to divide the output for 255
    #return revert_preprocessing(X) / 255

noise_scale = (input_range[1]-input_range[0]) * 0.1
ri = input_range[0]  # reference input


# Configure analysis methods and properties
methods = [
    # NAME                    OPT.PARAMS                POSTPROC FXN               TITLE

    # Show input
    ("input",                 {},                       input_postprocessing,      "Input"),

    # Function
    ("gradient",              {"postprocess": "abs"},   mnistutils.graymap,        "Gradient"),
    ("smoothgrad",            {"noise_scale": noise_scale,
                               "postprocess": "square"},mnistutils.graymap,        "SmoothGrad"),

    # Signal
    ("deconvnet",             {},                       mnistutils.bk_proj,        "Deconvnet"),
    ("guided_backprop",       {},                       mnistutils.bk_proj,        "Guided Backprop",),
    ("pattern.net",           {"pattern_type": "relu"}, mnistutils.bk_proj,        "PatternNet"),

    # Interaction
    ("pattern.attribution",   {"pattern_type": "relu"}, mnistutils.heatmap,        "PatternAttribution"),
    ("deep_taylor.bounded",   {"low": input_range[0],
                               "high": input_range[1]}, mnistutils.heatmap,        "DeepTaylor"),
    ("input_t_gradient",      {},                       mnistutils.heatmap,        "Input * Gradient"),
    ("integrated_gradients",  {"reference_inputs": ri}, mnistutils.heatmap,        "Integrated Gradients"),
    ("lrp.z",                 {},                       mnistutils.heatmap,        "LRP-Z"),
    ("lrp.epsilon",           {"epsilon": 1},           mnistutils.heatmap,        "LRP-Epsilon"),
    ("lrp.sequential_preset_a_flat",{"epsilon": 1},     mnistutils.heatmap,       "LRP-PresetAFlat"),
    ("lrp.sequential_preset_b_flat",{"epsilon": 1},     mnistutils.heatmap,       "LRP-PresetBFlat"),
    ("deep_lift.wrapper",     {"reference_inputs": ri}, mnistutils.heatmap,        "DeepLIFT Wrapper - Rescale"),                                                                 
    ("deep_lift.wrapper",     {"reference_inputs": ri, "nonlinear_mode": "reveal_cancel"}, mnistutils.heatmap,        "DeepLIFT Wrapper - RevealCancel")
]


# The main loop below will now instantiate the analyzer objects based on the loaded/trained model and the analyzers' parameterizations above.


# Create model without training softmax
model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

# Create analyzers.
analyzers = []
for method in methods:
    analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                            model_wo_softmax, # model without softmax output
                                            **method[1])      # optional analysis parameters

    # Some analyzers require training.
    analyzer.fit(data[0], batch_size=8, verbose=1)
    analyzers.append(analyzer)

##Extract the analyzer for the test set, EEG test image and the test label
n = 4
test_images = list(zip(data[2][:n], data[3][:n]))

print(test_images)

analysis = np.zeros([len(test_images), len(analyzers), 752, 30, 1])
text = []

str1=str(sys.argv[1])
str2=str(sys.argv[2])
strn1=str1.split('_')
strn2=strn1[8].split('.') ## change it for 10 or 9 depending if the folder name has an extra _

for i, (x, y) in enumerate(test_images):
    # Add batch axis.
    x = x[None, :, :, :]
    # Predict final activations, probabilites, and label.
    presm = model_wo_softmax.predict_on_batch(x)[0]
    prob = model.predict_on_batch(x)[0]
    y_hat = prob.argmax()
    # Save prediction info:
    text.append(("%s" % label_to_class_name[y],    # ground truth label
                 "%.2f" % presm.max(),             # pre-softmax logits
                 "%.2f" % prob.max(),              # probabilistic softmax output  
                 "%s" % label_to_class_name[y_hat] # predicted label
                ))

    for aidx, analyzer in enumerate(analyzers):
        # Analyze.
        a = analyzer.analyze(x)

        # Apply common postprocessing, e.g., re-ordering the channels for plotting.
        a = mnistutils.postprocess(a)
        # Apply analysis postprocessing, e.g., creating a heatmap.
        a = methods[aidx][2](a)
        # Store the analysis.
        analysis[i, aidx] = a[0]

        print(analysis[i,aidx],aidx) 
        if not os.path.exists("/dir_address/innvestigate_results/"+strn1[7]): ## change the str index depending on the name an structure of your folder
             os.makedirs("/dir_address/innvestigate_results/"+strn1[7])
        np.savetxt("method_"+str(aidx)+".txt",np.squeeze(np.asarray(analysis[i,aidx],dtype=np.int64)),delimiter=",")
