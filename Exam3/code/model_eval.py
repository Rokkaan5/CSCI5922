'''
Code to automate some of the repetitive processes/requirements.
'''
# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from data_preprocess import X_train,X_test,y_train,y_test, OHE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

# %%
class exam_model_eval:
    def __init__(self,NN_model,
                 model_name = 'NN Model',
                 X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test,
                 OHE=OHE):
        self.NN_model = NN_model
        self.model_name = model_name
        self.OHE = OHE
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.labeled_y_train = self.OHE.inverse_transform(self.y_train)
        self.y_test = y_test
        self.labeled_y_test = self.OHE.inverse_transform(self.y_test)
        
        

        # print model summary
        print(self.NN_model.summary())


    def compile_model(self, 
                      loss = tf.keras.losses.CategoricalCrossentropy(),
                      metrics = keras.metrics.CategoricalAccuracy(),
                      optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.001)):
        self.NN_model.compile(loss=loss,
                              metrics = metrics,
                              optimizer = optimizer)
        print("Model was compiled (using following parameters):")
        print("Loss Function =",str(loss))
        print("Accuracy Metric =",str(metrics))
        print("Optimizer =",str(optimizer))
        
        

    def train_model(self,epochs = 10):
        self.hist = self.NN_model.fit(self.X_train,
                                         self.y_train,
                                         epochs = epochs,
                                         validation_data =(self.X_test,self.y_test))

    