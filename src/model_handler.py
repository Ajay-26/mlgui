import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui
from nicegui.events import UploadEventArguments
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import os

class ModelHandle:
    """
        This class will handle all model related matters.
        The ui allows the user to click an "add layer" button which will call the add_layer function
        For now, we use the keras sequential api but can move to another format.
        As a result, we only have access to a few types of layers.
    """
    def __init__(self):
        self.layer_list = []
        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.loss_fn = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        
    def add_layer(self,layer_type : str, layer_param):
        if layer_type == "dense":
            #Dense layer, param would be the size of the output
            self.layer_list.append(layers.Dense(layer_param))
        elif layer_type == "activation":
            #Activation layer, param would be type of activation
            self.layer_list.append(layers.activation(layer_param))
    
    def set_lr(self,lr: float):
        self.lr = lr
    
    def set_optimizer(self,opt: str):
        if opt == "adam":
            self.optimizer = tf.keras.optimizers.Adam(self.lr)        
    
    def set_loss_fn(self,loss_fn: str):
        if loss_fn == "bce":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()  
        elif loss_fn == "bce_logits":
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits = True)
            
    def set_data(self, x,y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,y,test_size=0.2)
    
    def plot_training_outputs(self,history):
        metrics = history.history
        
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch')
        plt.ylabel('Loss [CrossEntropy]')
        plt.savefig("training_losses.png")
        plt.clf()

        plt.subplot(1,2,2)
        plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
        plt.legend(['accuracy', 'val_accuracy'])
        plt.ylim([0, 100])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy [%]')
        plt.savefig("training_accuracy.png")
        plt.clf()

        #Files can be deleted at this point because they are saved in mlflow
        os.remove("training_losses.png")
        os.remove("training_accuracy.png")    
            
    def train_model(self):
        self.ml_model = keras.models.Sequential(self.layer_list)
        ui.markdown(self.ml_model.summary())
        self.ml_model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=['accuracy']) 
        self.history = self.ml_model.fit(self.x_train,self.y_train, validation_split=0.2,epochs=50,batch_size=32)
        
    def display_results(self):
        ui.markdown(self.history)
        self.ml_model.evaluate(self.x_test,self.y_test)
        self.plot_training_outputs(self.history)
        ui.image("training_loss.png")
        ui.image("training_accuracy.png")
        