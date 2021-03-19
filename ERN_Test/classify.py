#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:48:08 2021

@author: louisbard
"""


"""
 Using EEGNet to classify P300 EEG data, using the sample dataset provided in:
     https://www.kaggle.com/c/inria-bci-challenge/data
   
 The two classes used from this dataset are:
     0: Bad Feedback, when the selected item is different from the expected item.
     1: Good Feedback, when the selected item is similar to the expected item.
"""

import numpy as np 

import pandas as pd 

# EEGNet-specific imports

from EEGModels import EEGNet

from tensorflow.keras import utils as np_utils

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import backend as K


K.set_image_data_format('channels_last')

# Sklearn metrics

from sklearn.metrics import roc_auc_score


##################### Import, process data for model ######################

train_labels = pd.read_csv('./data/TrainLabels.csv')

X_train_valid = np.load('./preproc/epochs.npy')

X_train_valid = np.reshape(X_train_valid, (16*340, 56,260))

y_train_valid = train_labels['Prediction'].values

X_test = np.load('./preproc/test_epochs.npy')

X_test = np.reshape(X_test,(3400,56,260))

y_test = np.reshape(pd.read_csv('./data/true_labels.csv', header=None).values, 3400)

# Partition

X_train = X_train_valid[1360:,:]

X_valid = X_train_valid[:1360,:]

Y_train = y_train_valid[1360:]

Y_valid = y_train_valid[:1360]

# Data contains 56 channels and 260 time-points. Set the number of kernels to 1.

kernels, chans, samples = 1, 56, 260






# Convert data to NCHW (kernels, channels, samples) format. 

X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)

X_valid   = X_valid.reshape(X_valid.shape[0], chans, samples, kernels)

X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print(str(X_train.shape[0]) + ' train samples')

print(str(X_valid.shape[0]) + ' validation samples')

print(str(X_test.shape[0]) + ' test samples')



############################# EEGNet ##################################

#kernels, chans, samples = 1, 60, 151

# Configure the EEGNet-8,2,16 model with kernel length of 260 samples 
model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# Compile the model and set the optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# Set a valid path for system to record model checkpoints
filepath = 'best_weights_bwbs_2.hdf5'
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                               save_best_only=True)

###############################################################################
# Since the classification task is imbalanced (significantly more trials in one
# class versus the others) can assign a weight to each class during 
# optimization to balance it out.
###############################################################################

# Syntax is {class_1:weight_1, class_2:weight_2,...}.
 
# Weighted loss
weight_0 = 1/(len([y for y in y_train_valid if y == 0]))
weight_1 = 1/(len([y for y in y_train_valid if y == 1]))

class_weights = {0:weight_0, 1:weight_1}

################################################################################
# Fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run.
################################################################################
fittedModel = model.fit(X_train, Y_train, batch_size = 34, epochs = 100, 
                        verbose = 2, validation_data=(X_valid, Y_valid),
                        callbacks=[checkpointer], class_weight = class_weights)

# Load optimal weights

filepath = 'best_weights_bwbs_2.hdf5'


model.load_weights(filepath)

###############################################################################
# Make prediction on test set.
###############################################################################

probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == y_test.argmax(axis=-1))




auc         = roc_auc_score(y_test,preds)




print("Classification Accuracy: %f " % (acc))
print("Area Under Curve: %f" % (auc))





