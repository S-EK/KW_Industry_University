# -*- coding: utf-8 -*-
"""ResNext_Module.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yzwuxcoDnGayWfaTRzQFsjIs7ZV-n7a-
"""

#0. Import libraries, Download pretrained model
import keras
import kerastuner as kt
import keras_tuner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from classification_models.tfkeras import Classifiers
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import *

class myResNext:
  def __init__(self, X, y, lzClass, iBatch, iEpoch, iRandom_seed):
    self.X= X
    self.y= y
    self.lzClass= lzClass
    self.iEpoch= iEpoch
    self.iBatch= iBatch
    self.iRandom_seed = iRandom_seed


  def preprocessing(self):
    #1. flatten
    self.X= self.X.astype('float32')
    self.X= tf.keras.applications.resnet50.preprocess_input(self.X)

    #2. data split
    self.X_train, self.X_test, self.y_train, self.y_test \
     = train_test_split(self.X, self.y, test_size=int(len(self.X)*0.4), random_state=self.iRandom_seed)
    self.X_valid, self.X_test, self.y_valid, self.y_test \
     = train_test_split(self.X_test, self.y_test, test_size=0.5)


  def model_resnext(self, hp):
    #3. base model generation
    ResNeXt50, preprocess_input = Classifiers.get('resnext50')
    base = ResNeXt50(include_top = False, input_shape=(224, 224, 3), weights='imagenet')
    base.trainable= False

    lzDrop= [.2, .3, .5]
    lzLR= [.01, .005, .001, .0005, .0001]
    lzOpt= ['sgd', 'adam']

    inputs = Input(shape = (32, 32, 3))
    x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(inputs)
    x = base(x, training = False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(hp.Choice('dropout', lzDrop, default= .5))(x)
    outputs = tf.keras.layers.Dense(self.lzClass, activation = 'softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    opt= keras.optimizers.get(hp.Choice('optimizer', lzOpt, default= 'sgd'))
    opt.learning_rate= hp.Choice('learning_rate', lzLR, default= .001)
    model.compile(optimizer = opt, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

    return model


  def inference(self):
    #4. fine tuning
    self.tuner = kt.Hyperband(hypermodel = self.model_resnext,
                     objective = 'val_accuracy', 
                     max_epochs = self.iEpoch,
                     directory = 'resNext',
                     project_name = 'intro_to_kt',
                     factor= 3)
    
    self.tuner.search(self.X_train, self.y_train, epochs=self.iEpoch,
        validation_data=(self.X_valid, self.y_valid),
        batch_size= self.iBatch)
    

  def predictResult(self):
    self.result = []
    self.best_hps = self.tuner.get_best_hyperparameters(5)

    for i in range(0, 5):
      best_hp = self.best_hps[i]
      es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3)

      model = self.model_resnext(best_hp)

      model_result = model.fit(self.X_train, self.y_train,
                                    batch_size=self.iBatch,
                                    epochs=self.iEpoch,
                                    validation_data=(self.X_test, self.y_test),
                                    callbacks=[es])
    
      start_time = time.time()
      y_pred = model.predict(self.X_test)
      inference_time = (time.time() - start_time) / 10

      y_pred = np.argmax(y_pred, axis=1)
    
      f1 = f1_score(self.y_test, y_pred, average='weighted')
      accuracy = accuracy_score(self.y_test, y_pred)

      self.result.append({
          'model' : 'resnext',
          'number' : i,
          'history' : {
              'accuracy': model_result.history['accuracy'],
              'val_accuracy' :model_result.history['val_accuracy'],
              'loss': model_result.history['loss'],
              'val_loss' :  model_result.history['val_loss'],
          },
          'f1' : f1,
          'inference_time' : inference_time,
          'accuracy' : accuracy,
          'best_parameter' : {
              'dropout':best_hp.get('dropout'),
              'optimizer':best_hp.get('optimizer'),
              'learning_rate':best_hp.get('learning_rate'),
          }
      })

    return self.result


  def getModel(self, num):
    model = self.model_vgg16(self.best_hps[num])
    return model