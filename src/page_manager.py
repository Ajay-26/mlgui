from nicegui import ui
from nicegui.events import UploadEventArguments
from data_handler import DataSetHandle
from model_handler import ModelHandle

class PageManager:
    """
        This is a class which consists some metadata as well as handles for all other aspects of the ui
        It provides a means of communication for the dataset handle and the model handle
    """
    def __init__(self):
        self.dataset_handler = DataSetHandle()
        self.model_handler = ModelHandle()
        self.data_recv = True
        
    def receive_data_upload(self,e : UploadEventArguments, page_redirect):
        self.dataset_handler.initialise_data(e)
        self.data_recv = False
        ui.open(page_redirect)    
        
    def choose_label(self,model_redirect):
        cols = self.dataset_handler.get_columns()
        print(cols)
        selection = ui.select(cols,value = cols[-1])
        #print(selection.value)
        self.dataset_handler.choose_column(selection.value)
        x,y = self.dataset_handler.get_data()
        self.model_handler.set_data(x,y)
        ui.button("Go",on_click=lambda e: ui.open(model_redirect))
        
    def display_model_ui(self):
        selection = ui.select(self.model_handler.layer_options)
        param = ui.input("Enter model parameter")
        ui.button("Add Layer", on_click=lambda e: self.model_handler.add_layer(selection.value, param.value))
        lr_input = ui.input("Enter learning rate")
        loss_fn_input = ui.select(self.model_handler.loss_fn_options)
        optimizer_input = ui.select(self.model_handler.optimizer_options)
        ui.button("Create Model", on_click=lambda e: self.model_handler.train_model(lr_input.value, loss_fn_input.value, optimizer_input.value))