import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui
from nicegui.events import UploadEventArguments
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ModelHandle:
    """
        This class will handle all model related matters.
        The ui allows the user to click an "add layer" button which will call the add_layer function
        For now, we use the keras sequential api but can move to another format.
        As a result, we only have access to a few types of layers.
    """
    def __init__(self):
        self.layer_list = []
        
    def add_layer(self,layer_type : str, layer_param):
        if layer_type == "dense":
            #Dense layer, param would be the size of the output
            self.layer_list.append(layers.Dense(layer_param))
        elif layer_type == "activation":
            #Activation layer, param would be type of activation
            self.layer_list.append(layers.activation(layer_param))
        
