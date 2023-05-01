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
        self.layer_metadata = []
        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.loss_fn = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.training_loss_png_path = "training_loss.png"
        self.training_acc_png_path = "training_loss.png"
        self.layer_options = ["Dense","Activation"]
        self.optimizer_options = ["adam"]
        self.loss_fn_options = ["bce","bce_logits"]
        self.model_graphic = None
        
    def show_model_graphic(self):
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        edge_str = ""
        print(self.layer_metadata)
        num_layers = len(self.layer_metadata)
        for i in range(num_layers):
            if i < num_layers - 1:
                j = i+1
                edge_str = edge_str + f"{alphabet[i]}[{self.layer_metadata[i]}] -->  {alphabet[j]}[{self.layer_metadata[j]}];\n"
        mermaid_str = f'''
        flowchart LR;
            {edge_str}
        '''
        print(mermaid_str)
        if self.model_graphic is None:
            self.model_graphic = ui.mermaid(mermaid_str)
        else:
            self.model_graphic.delete()
            self.model_graphic = ui.mermaid(mermaid_str)    
    def add_layer(self,layer_type : str, layer_param):
        print(layer_type, layer_param)
        if not (layer_type in self.layer_options):
            ui.notify("This layer isn't available")
            return
        if layer_type == "Dense":
            if not layer_param.isnumeric():
                ui.notify("Wrong layer and parameter combination")
                return
            #Dense layer, param would be the size of the output
            self.layer_list.append(layers.Dense(layer_param))
            self.layer_metadata.append(f"Dense-{layer_param}")
        elif layer_type == "Activation":
            #Activation layer, param would be type of activation
            if not layer_param.isalpha():
                ui.notify("Wrong layer and parameter combination")
                return
            self.layer_list.append(layers.Activation(layer_param))
            self.layer_metadata.append(f"Activation-{layer_param}")
        self.show_model_graphic()
        print("Perhaps displayed model graphic ?")
    
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
        self.input_shape = self.x_train.shape[1:]
    
    def plot_training_outputs(self,history):
        metrics = history.history
        
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1)
        plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
        plt.legend(['loss', 'val_loss'])
        plt.ylim([0, max(plt.ylim())])
        plt.xlabel('Epoch')
        plt.ylabel('Loss [CrossEntropy]')
        plt.savefig(self.training_loss_png_path)
        plt.clf()

        plt.subplot(1,2,2)
        plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
        plt.legend(['accuracy', 'val_accuracy'])
        plt.ylim([0, 100])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy [%]')
        plt.savefig(self.training_acc_png_path)
        plt.clf()  
            
    def train_model(self, loss_fn_input: str, lr_input: str, optimizer_input: str):
        print(loss_fn_input, lr_input, optimizer_input)
        if len(self.layer_list) <= 0:
            ui.notify("You haven't added any layers, please add layers")
        try:
            if not (loss_fn_input in self.loss_fn_options):
                ui.notify("This loss function is not available")
            if not lr_input.replace(".","").isnumeric():
                ui.notify("Invalid format for learning rate input")
            if not (optimizer_input in self.optimizer_options):
                ui.notify("This optimizer is not available")
        except Exception as e:
            ui.notify("Please enter in all the fields")
        try:        
            self.set_loss_fn(loss_fn_input)
            self.set_optimizer(optimizer_input)
            self.set_lr(float(lr_input))
        except Exception as e:
            ui.notify(f"Exception occured while setting model parameters: {e}")
        self.ml_model = keras.models.Sequential([layers.Input(self.input_shape)] + self.layer_list)
        self.ml_model.build(self.input_shape)
        stringlist = []
        self.ml_model.summary(print_fn=lambda x: stringlist.append(x))
        ui.markdown(stringlist)
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
        