# -*- coding: utf-8 -*-
"""VGG16_Module.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15IrUcyTvbY_vGyNNtE0_GRf3CTwv6SmC
"""

# Import library
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

import keras_tuner
import kerastuner as kt

from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class ModuleVGG16:
  def __init__(self, X, y, num_class, img_dim, batch_size, epochs, random_seed):
    self.num_class = num_class
    self.img_dim = img_dim
    self.batch_size = batch_size
    self.epochs = epochs

    self.random_seed = random_seed

    self.X = X
    self.y = y
    
  
  # Split and astype: Train:Valid:Test = 6:2:2
  def preprocessing(self):
    self.X = self.X.astype(float)/255

    self.X_train, self.X_test, self.y_train, self.y_test \
     = train_test_split(self.X, self.y, test_size=int(len(self.X)*0.4), random_state=self.random_seed)
    self.X_valid, self.X_test, self.y_valid, self.y_test \
     = train_test_split(self.X_test, self.y_test, test_size=0.5)


  # Define VGG16 Module with Transfer Learning
  def model_vgg16(self, hp):
    dropout = hp.Choice('dropout', [0.2, 0.3, 0.5], default=0.2)

    pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=self.img_dim)
    pre_trained_vgg.trainable = False

    model = keras.Sequential()
    model.add(pre_trained_vgg)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(self.num_class, activation='softmax'))

    optimizer_name = hp.Choice('optimizer', ['adam', 'sgd'], default='adam')
    optimizer = keras.optimizers.get(optimizer_name)
    optimizer.learning_rate = hp.Choice(
                  'learning_rate', [0.01, 0.005, 0.001, 0.0005, 0.0001], default=0.01)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


  # Hyperparameter tuning with keras tuner
  def inference(self):
    checkpoint = ModelCheckpoint(filepath='pretrained_VGG16', 
        monitor='loss', 
        mode='min', 
        save_best_only=True)

    self.tuner = kt.Hyperband(
        self.model_vgg16,
        objective ='val_accuracy',
        max_epochs =self.epochs,
        directory = 'vgg16',
        project_name = 'intro_to_kt')
    
    self.tuner.search(self.X_train, self.y_train, epochs=self.epochs,
        validation_data=(self.X_valid, self.y_valid),
        batch_size= self.batch_size,
        callbacks = [checkpoint])

  
  # Calculate evaluation metrics
  def predictResult(self):
    self.result = []
    self.best_hps = self.tuner.get_best_hyperparameters(5)


    for i in range(0, 5):
      best_hp = self.best_hps[i]
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience = 1)

      model = self.model_vgg16(best_hp)

      model_result = model.fit(self.X_train, self.y_train,
                                    batch_size=self.batch_size,
                                    epochs=self.epochs,
                                    validation_data=(self.X_test, self.y_test),
                                    callbacks=[early_stopping])
    
      start_time = time.time()
      y_pred = model.predict(self.X_test)
      inference_time = (time.time() - start_time) / 10

      y_pred = np.argmax(y_pred, axis=1)
    
      f1 = f1_score(self.y_test, y_pred, average='weighted')
      accuracy = accuracy_score(self.y_test, y_pred)

      self.result.append({
          'model' : 'vgg16',
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


  # Return Selected Model
  def getModel(self, num):
    model = self.model_vgg16(self.hps[num])
    return model