""" Convolutional Neural Network (CNN)
Build and train a convolutional neural network with TensorFlow for EEG-based emotion decoding.
Author: Aymeric Damien
Modified by: Juan Manuel Mayor-Torres
"""

from __future__ import division, print_function, absolute_import

import os
import logging
from subprocess import Popen, PIPE
import numpy as np
#from sklearn import datasets, metrics
import tensorflow as tf

import sys
from numpy import genfromtxt

str1 = str(sys.argv[1])
str2 = str(sys.argv[2])
strn1 = str1.split('_')
# change it for 10 or 9 depending if the folder name has a _ please check the length of your pathname and the number of _s it contains
strn2 = strn1[8].split('.')

# training and test files taking per subject specifiy these files in the LOTO CV in an upper level
csvFile1 = str(sys.argv[1])
csvFile2 = str(sys.argv[2])
# path_for_method_mask_provided_by_iNNvestigate ## refer to some mask examples provided by the
csv_method = "/path_preffix/"+strn1[7] + "/method_"+strn2[0]+"_"+str(sys.argv[4])+".txt"


# set up the usage of multiple GPUs by your own in tensorflow you can modify this code depending on your suitability
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cwd = os.getcwd()

# here you can change your checkpoint and use only one and different for each cross-validation you run
fileList = os.listdir(cwd+'/folder_check_points_innvestigate')
for fileName in fileList:
    if fileName != 'eval':
        os.remove(cwd+'/folder_check_points_innvestigate'+"/"+fileName)

if os.path.exists(cwd+'/folder_check_points_innvestigate/eval'):
    fileList2 = os.listdir(cwd+'/folder_check_points_innvestigate/eval')
    for fileName in fileList2:
        os.remove(cwd+'/folder_check_points_innvestigate/eval'+"/"+fileName)


tf.logging.set_verbosity(tf.logging.INFO)

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.INFO)

# create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

os.remove(cwd+'/file_report_innvestigate.txt')
# create file handler which logs even debug messages
fh = logging.FileHandler(cwd+'/file_report_innvestigate.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
log.addHandler(fh)


# reading training and test files
# organizing tuples for coding evaluation

str1 = str(sys.argv[1])
str2 = str(sys.argv[2])
strn1 = str1.split('_')

training_data_file = genfromtxt(csvFile1, delimiter=',', skip_header=0)
test_data_file = genfromtxt(csvFile2, delimiter=',', skip_header=0)
method_file = genfromtxt(csv_method, delimiter=',', skip_header=0)
data_shape = "There are " + \
    repr(training_data_file.shape[0]) + \
    " samples of vector length " + repr(training_data_file.shape[1])
method_shape = "There are " + \
    repr(method_file.shape[0]) + \
    " samples of vector length " + repr(method_file.shape[1])
num_rows = training_data_file.shape[0]  # Number of data samples
num_cols = training_data_file.shape[1]  # Length of Data Vector

num_rowst = 1  # Number of data samples
# test_data_file.shape[1] # Length of Data Vector
num_colst = test_data_file.shape[0]

num_rowsm = method_file.shape[0]  # Number of data samples
# test_data_file.shape[1] # Length of Data Vecto
num_colsm = method_file.shape[1]

total_size_train = (num_cols-1)*num_rows
total_size_test = (num_colst-1)*num_rowst
total_size_m = (num_colsm)*num_rowsm

data_train = np.arange(total_size_train)
data_train = data_train.reshape(
    num_rows, num_cols-1)  # 2D Matrix of data points
data_train = data_train.astype('float64')

label_train = np.arange(num_rows)
label_train = label_train.astype('int32')

data_test = np.arange(total_size_test)
data_test = data_test.reshape(
    num_rowst, num_colst-1)  # 2D Matrix of data points
data_test = data_test.astype('float64')

label_test = np.arange(num_rowst)
label_test = label_test.astype('int32')

# define method mask variables
data_m = np.arange(total_size_m)
data_m = data_m.reshape(num_rowsm, num_colsm)  # 2D Matrix of data points
data_m = data_m.astype('float64')


# Align data from the csv file, read through data_train file, assume label is in last col following the instructions of the README file to create the input training and test files


for i in range(training_data_file.shape[0]):
    label_train[i] = training_data_file[i][num_cols-1]
    for j in range(num_cols-1):
        data_train[i][j] = training_data_file[i][j]

# Read through data_test file, assume label is in last col as we explain on the README file please create all files as a csv file with a comma delimiter

for i in range(1):
    label_test[i] = test_data_file[num_colst-1]
    for j in range(num_colst-1):
        data_test[i][j] = test_data_file[j]

# method section

for i in range(method_file.shape[0]):
    #label_train[i] = method_file[i][num_colsm-1]
    for j in range(num_colsm-1):
        data_m[i][j] = method_file[i][j]

#data_m[data_m > 1] = 1.0

# remove the more important features depending on the rate removal value
data_m[data_m > 256] = 100
data_m = (data_m-data_m.min())/(data_m.max()-data_m.min())

# change this parameter between 0-1 and check the mask and corresponding performance output a different r value produces
pos_f = 0.5*(data_m.shape[0]*data_m.shape[1])
i_sorted = np.argsort(data_m, axis=None)[::-1]
rows_i, col_i = np.unravel_index(i_sorted, data_m.shape)
for i in range(len(rows_i)):
    if i < pos_f:
        data_m[rows_i[i], col_i[i]] = 1.0
    else:
        data_m[rows_i[i], col_i[i]] = 0.0

# Training Parameters - change them if you want to tune the CNN differently
learning_rate = 0.00001
num_steps = 300
batch_size = 4

# Network Parameters - change them if you want to tune the CNN differently.
num_input = 22560  # 752*30 time-points*channels input burst.
num_classes = 4  # 4 emotions classes from DANVA-2 happy, sad, angry and fear.
dropout = 0.25  # Dropout, probability to drop a in the final dense layer.

# Creating the neural network and variables


def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # Data input is a 1-D vector of 22560 features (752*30 pixels)
        # Reshape to match picture format [1 x Time Points x Channels x 1]
        # Tensor input become 4-D: [Batch Size, 1 , Time Points, Channels]
        #x = tf.reshape(x, shape=[-1, 752, 30, 1])

        # Convolution Layer with 32 filters and a kernel size of 100x10
        conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[
                                 100, 10], activation=tf.nn.relu, name='conv1d')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, pool_size=[5, 2], strides=2)
        # add the normalization if you like or not is not necessary for the final performance achieving
        tf.layers.LayerNormalization(axis=1, center=True, scale=True)
        # Convolution Layer with 64 filters and a kernel size of 20x5
        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=[
                                 20, 5], activation=tf.nn.relu, name='conv2d')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
        # add the normalization if you like or not is not necessary for the final performance achieving
        tf.layers.LayerNormalization(axis=1, center=True, scale=True)
        # Convolution Layer with 128 filters and a kernel size of 5x2
        conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=[
                                 10, 2], activation=tf.nn.relu, name='conv3d')
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)

        # use these tensors for inner model exploration in the outter session
        kernel = [v for v in tf.trainable_variables() if v.name ==
                  "ConvNet/conv1d/kernel:0"]
        bias = [v for v in tf.trainable_variables() if v.name ==
                "ConvNet/conv1d/bias:0"]

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv3)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out, kernel, bias


# Define the model function (following TF Estimator Template) ## model definition adequated for being for the emotion decoding
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    datn_train, kernel_train, bias_train = conv_net(features, num_classes, dropout, reuse=False,
                                                    is_training=True)
    datn_test, kernel_test, bias_test = conv_net(features, num_classes, dropout, reuse=True,
                                                 is_training=False)

    #kernel_ten = tf.convert_to_tensor(kernel_train, name="kernel_tensor")
    # Predictions
    pred_classes = tf.argmax(datn_test, axis=1, name="argmax_tensor")
    # for predicting probabilities we use softmax and for predicting emotion labels outputs argmax but both can be used in the same way
    #pred_probas = tf.nn.softmax(datn_test, name="softmax_tensor")

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=datn_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(
        labels=labels, predictions=pred_classes, name="labels")
    #training_summary = tf.convert_to_tensor(acc_op, name="training_accuracy")

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# before use the functions above the data from the CSV files should be converted in float32 to be processed by the pipeline
data_train = data_train.astype('float32', casting='same_kind')
# labels should be defined between 0-3
label_train = label_train.astype('float32', casting='same_kind')-1
data_test = data_test.astype('float32', casting='same_kind')
label_test = label_test.astype('float32', casting='same_kind')-1

# Build the Estimator
model = tf.estimator.Estimator(
    model_fn, cwd+'/folder_check_points_innvestigate')

# use this tensor template to log into the inner model for debugging
#tensors_to_log = {"probabilities": "softmax_tensor" , "probabilities2": "argmax_tensor", "train_acc":"training_accuracy"}
tensors_to_log = {"probabilities": "softmax_tensor", "probabilities2": "argmax_tensor",
                  "train_acc": "training_accuracy", "kernetl_t": "kernel_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=1)
logging_hook_t = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_secs=1)

# reshape the training data for weighting features
data_tr = data_train.reshape((-1, 752, 30))

# features re-training multiplying directly with the binary mask
for k in range(47):
    data_tr[k, :, :] = np.multiply(data_tr[k, :, :], data_m)

# convert to float32 for training
data_tr = data_tr.astype('float32')
data_tr = data_train.reshape((-1, 752, 30, 1))

# Define the input function for training

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': data_tr}, y=label_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# Train the Model with the features occluded by the method pattern
mtrain = model.train(input_fn, steps=num_steps, hooks=[logging_hook])

# Evaluate the Model
# Define the input function for evaluating

data_t = data_test.reshape((-1, 752, 30, 1))
data_t = np.multiply(data_t.reshape((752, 30)), data_m)
data_t = data_t.astype('float32')
data_t = data_t.reshape((-1, 752, 30, 1))

# Evaluate the Model
# Define the input function for evaluating
input_pn = tf.estimator.inputs.numpy_input_fn(
    x={'images': data_t}, y=label_test,
    batch_size=batch_size, shuffle=False)

e = model.evaluate(input_pn, hooks=[logging_hook_t])
print("Testing Accuracy:", e['accuracy'])
# writing the report per each trial
q = Popen('/bin/sh comp_res_abs.sh.sh "%s" "%s"' % (strn1[7]+"_test_"+str(sys.argv[4]), strn2[0]), stdin=PIPE, stdout=PIPE,
          shell=True)
out = q.stdout.read()
print(out)
