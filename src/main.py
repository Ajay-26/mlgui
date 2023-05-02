from nicegui import ui
import numpy as np
import pandas as pd
from page_manager import PageManager

x = np.linspace(0,1,100)
y = np.sin(x)
df = pd.DataFrame(np.concatenate((x,y),axis=0))
df.to_csv('test_file.csv')

@ui.page('/')
def index_page():
    ui.markdown("Dataset options:")
    ui.upload(on_upload=lambda e: page_manager.receive_data_upload(e,dataset_page)).classes('max-w-full')
    
    
@ui.page('/dataset')
def dataset_page():
    ui.markdown("Choose the column you want as labels")
    page_manager.choose_label(model_page)    
    
@ui.page('/model')
def model_page():
    page_manager.display_model_ui()

global page_manager

page_manager = PageManager()

ui.run(port=8080)