# -*- coding: utf-8 -*-
"""Densenet_Module.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MYzunbfOmBUBjiWzH2iTMjlILUn_PhCi
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import train_test_split
import IPython
from keras.applications.densenet import DenseNet121
from keras import Input, Model
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD, Adam
import time
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)

class modelDensenet:
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

    self.x_train, self.x_test, self.y_train, self.y_test \
     = train_test_split(self.X, self.y, test_size=int(len(self.X)*0.4), random_state=self.random_seed)
    self.x_train, self.x_valid, self.y_train, self.y_valid \
      = train_test_split(self.x_train, self.y_train, test_size=0.2, shuffle=True)


  def model_densenet(self, hp):
    base_model = DenseNet121(weights = 'imagenet', input_shape = self.img_dim, pooling = 'avg',
                        include_top = False, classes = self.num_class)
    base_model.trainable = False

    inputs = Input(shape = self.img_dim)
    x = base_model(inputs, training = False)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(hp.Choice('dropout', [0.2, 0.3, 0.5]))(x)
    outputs = Dense(10, activation = 'softmax')(x)

    model_densenet = Model(inputs, outputs)

    hp_learning_rate = hp.Choice('learning_rate', values = [0.01, 0.005, 0.001, 0.0005, 0.0001])
    hp_optimizer = hp.Choice('optimizer', values=['sgd', 'adam'])

    if hp_optimizer == 'sgd':
        optimizer = SGD(learning_rate=hp_learning_rate)
    elif hp_optimizer == 'adam':
        optimizer = Adam(learning_rate=hp_learning_rate)
    else:
        raise

    model_densenet.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    
    return model_densenet
    

  def inference(self):
    self.tuner = kt.Hyperband(self.model_densenet,
                     objective = 'val_accuracy', 
                     max_epochs = self.epochs,
                     factor = 3,
                     directory = 'densenet',
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

      model = self.model_densenet(best_hp)
      
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
    model = self.model_vgg16(self.hps[num])
    return model