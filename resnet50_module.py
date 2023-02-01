# -*- coding: utf-8 -*-
"""Resnet50_Module.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1DLs4QnqeHtu2r7oGsxTVMXi0Uegl7FFn
"""

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
import warnings
from sklearn.model_selection import train_test_split
import keras
import kerastuner as kt
import numpy as np
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

class modelResnet50:
  def __init__(self, X, y, num_class, img_dim, batch_size, epochs, random_seed):
    self.num_class = num_class
    self.img_dim = img_dim
    self.batch_size = batch_size
    self.epochs = epochs
    
    self.random_seed = random_seed

    self.X = X
    self.y = y


  def preprocessing(self):
    input_images = self.X.astype('float32')

    self.X = tf.keras.applications.resnet50.preprocess_input(input_images)

    self.x_train, self.x_test, self.y_train, self.y_test \
     = train_test_split(self.X, self.y, test_size=int(len(self.X)*0.4), random_state=self.random_seed)
    self.x_train, self.x_valid, self.y_train, self.y_valid \
      = train_test_split(self.x_train, self.y_train, test_size=0.2, shuffle=True)


  def model_resnet50(self, hp):
    # 전이 학습
    base_model = ResNet50(weights = 'imagenet', input_shape = (224, 224, 3), 
                          include_top = False, classes = self.num_class)


    inputs = Input(shape = self.img_dim)
    x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(inputs)
    x = base_model(x, training = False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(hp.Choice('dropout', [0.2, 0.3, 0.5]))(x)
    outputs = Dense(10, activation = 'softmax')(x)
    model_res = tf.keras.Model(inputs, outputs)

    hp_learning_rate = hp.Choice('learning_rate', [0.01, 0.005, 0.001, 0.0005, 0.0001])
    hp_optimizer = hp.Choice('optimizer', values=['sgd'])

    if hp_optimizer == 'sgd':
      optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)

    else:
      raise
  
    model_res.compile(optimizer=optimizer,
      loss='sparse_categorical_crossentropy', 
      metrics=['accuracy'])
    
    return model_res
    

  def inference(self):
    self.tuner = kt.Hyperband(self.model_resnet50,
      objective='val_accuracy',
      max_epochs = self.epochs,
      directory = 'resnet50',
      project_name = 'intro_to_kt')

    self.tuner.search(self.x_train, self.y_train,
                        validation_data = (self.x_valid, self.y_valid),
                        epochs = self.epochs,
                        batch_size = self.batch_size)
    
  def predictResult(self):
    # 상위 5개 
    self.result = []
    self.best_hps = self.tuner.get_best_hyperparameters(5)

    for i in range(0, 5):
      best_hp = self.best_hps[i]
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience = 1)

      model = self.model_resnet50(best_hp)
      
      model_result = model.fit(self.x_train, self.y_train, validation_data = (self.x_valid, self.y_valid), \
                epochs=self.epochs, callbacks = [early_stopping])

      start_time = time.time()
      y_pred = model.predict(self.x_test)
      inference_time = (time.time() - start_time) / 10

      y_pred = np.argmax(y_pred, axis=1)
      
      f1 = f1_score(self.y_test, y_pred, average='weighted')
      accuracy = accuracy_score(self.y_test, y_pred)

      self.result.append({
          'model' : 'resnet50',
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