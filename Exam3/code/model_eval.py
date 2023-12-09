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
        self.labeled_y_train = self.OHE.inverse_transform(self.y_train) #likely not needed, but potentially useful
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

    def accuracy_plot(self,
                      title = "Accuracy over all Epochs"):
        # Accuracy plot
        plt.plot(self.hist.history['categorical_accuracy'], label='accuracy')
        plt.plot(self.hist.history['val_categorical_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title(title+" for {}".format(self.model_name))

        #plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

    def loss_plot(self,
                  title = "Loss over all epochs"):
        # Loss plot
        plt.plot(self.hist.history['loss'], label='loss')
        plt.plot(self.hist.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title+" for {} ".format(self.model_name))
        #plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

    def test_model(self,verbose=True):
        self.prediction = self.NN_model.predict(self.X_test)
        self.labeled_prediction = self.OHE.inverse_transform(self.prediction)
        self.eval_loss,self.eval_acc = self.NN_model.evaluate(self.X_test,self.y_test)
        
        if verbose:
            print("\nModel Loss from testing:")
            print(self.eval_loss)

            print("\nModel Accuracy from testing:")
            print(self.eval_acc)

            print("\nPreview of model prediction (raw):")
            print(self.prediction[:5])

            print("\nPreview of predictions labeled:")
            print(self.labeled_prediction[:5])

    def pretty_confusion_matrix(self,
                                title='Confusion Matrix'):

        cm = confusion_matrix(self.labeled_prediction,self.labeled_y_test)
        
        fig, ax = plt.subplots(figsize=(10,8)) 
        sns.heatmap(cm, annot=True, fmt='g', ax=ax, annot_kws={'size': 18})
        ax.set_xlabel('True labels') 
        ax.set_ylabel('Predicted labels')
        ax.xaxis.set_ticklabels(self.OHE.categories_[0].tolist())
        ax.yaxis.set_ticklabels(self.OHE.categories_[0].tolist())
        ax.set_title(title+" of {} Prediction Performance".format(self.model_name)) 
        plt.show()
