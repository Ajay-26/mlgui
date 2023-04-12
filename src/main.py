from nicegui import ui
import numpy as np
import pandas as pd
from page_manager import PageManager

x = np.linspace(0,1,100)
y = np.sin(x)
df = pd.DataFrame(np.concatenate((x,y),axis=0))
df.to_csv('test_file.csv')

#page_manager = PageManager()
ui.markdown("Dataset options:")
#ui.upload(on_upload=lambda e: page_manager.receive_data_upload(e)).classes('max-w-full')
#ui.input(label="Choose the column you want as labels").on('keydown.enter', lambda e: page_manager.choose_label(e.value))

ui.run(port=8080)