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
        
    def receive_data_upload(self,e : UploadEventArguments):
        self.dataset_handler.initialise_data(e)
        
    def choose_label(self,e: str):
        self.dataset_handler.choose_column(e)