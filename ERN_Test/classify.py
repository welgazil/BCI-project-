#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 22:48:08 2021

@author: louisbard
"""

#importation des modules nécessaires 

import numpy as np 

import pandas as pd 

#importation du réseau EEGNet et API Keras

from EEGModels import EEGNet

from tensorflow.keras import utils as np_utils

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import backend as K


K.set_image_data_format('channels_last')

# Sklearn metrique

from sklearn.metrics import roc_auc_score


##################### Importation des données prétraitées ######################

train_labels = pd.read_csv('./data/TrainLabels.csv')

X_train_valid = np.load('./preproc/epochs.npy')

X_train_valid = np.reshape(X_train_valid, (16*340, 56,260))

y_train_valid = train_labels['Prediction'].values

X_test = np.load('./preproc/test_epochs.npy')

X_test = np.reshape(X_test,(3400,56,260))

y_test = np.reshape(pd.read_csv('./data/true_labels.csv', header=None).values, 3400)


# Les données contiennent 56 canaux et 260 points 

kernels, chans, samples = 1, 56, 260


# Partition

X_test = X_train_valid[4080:]

y_test = y_train_valid[4080:]

X_train_valid= X_train_valid.reshape(X_train_valid.shape[0], chans, samples, kernels)

X_train_valid = X_train_valid[:4080]

y_train_valid = y_train_valid[:4080]


# Conversion au format NCHW  

X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)

X_valid   = X_valid.reshape(X_valid.shape[0], chans, samples, kernels)

X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print(str(X_train.shape[0]) + ' train samples')

print(str(X_valid.shape[0]) + ' validation samples')

print(str(X_test.shape[0]) + ' test samples')



############################# EEGNet ##################################



# Validation croisée à 4 plis 
kfold = KFold(n_splits=4, shuffle=True)

#
fold_no = 1
for train, test in kfold.split(X_train_valid,y_train_valid):

  print(X_train_valid[train])

  # Define the model architecture

  

 
  model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 100, F1 = 4, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

  # Compilation du modèle 
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

  # Sauvegarde des poids 
  filepath = 'best_weights_bwbs_2.hdf5'
  checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,
                               save_best_only=True)
  
  # Pondération en fonction de la répartition des classes 
 
  # Weighted loss
  weight_0 = 1/(len([y for y in y_train_valid if y == 0]))
  weight_1 = 1/(len([y for y in y_train_valid if y == 1]))

  class_weights = {0:weight_0, 1:weight_1}

################################################################################
# Fit du modèle
################################################################################
  fittedModel = model.fit(X_train_valid[train], y_train_valid[train], batch_size = 34, epochs = 100, 
                        verbose = 2, validation_data=(X_train_valid[test], y_train_valid[test]),
                        callbacks=[checkpointer], class_weight = class_weights)



  # 
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  
  # Evaluation de la performance sur l'ensemble test
  probs       = model.predict(X_train_valid[test])
  preds       = probs.argmax(axis = -1)  
  scores = model.evaluate(X_train_valid[test], y_train_valid[test], verbose=0)




  auc         = roc_auc_score(y_train_valid[test],preds)
  acc_per_fold.append(scores[1] * 100)
  auc_per_fold.append(auc)

  # On passe à un autre pli 
  fold_no = fold_no + 1
  

# Evaluation finale sur l'ensemble de test 
 
probs       = model.predict(X_test)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == y_test.argmax(axis=-1))




auc         = roc_auc_score(y_test,preds)




