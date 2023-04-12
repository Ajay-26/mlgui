from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nicegui import ui
from nicegui.events import UploadEventArguments
from io import StringIO

class DataSetHandle:
    """
        Used to handle all the data, will store the dataframe, as well as x and y. 
        The dataset for now needs to be in csv format, with columns being used. 
    """
    def __init__(self):
        self.style = "csv"
        self.df = None
        self.x = None
        self.y = None 
        
    def initialise_data(self,file : UploadEventArguments):
        data_list = []
        with file.content as f:
            content = f.read().decode()
        csvstring = StringIO(content)
        df = pd.read_csv(csvstring)
        self.df = df
        ui.notify(f"Columns are: {self.df.columns}")
            
    def choose_column(self, colname):
        y = self.df[colname]
        x = self.df.drop(colname)
        self.y = np.array(y)
        self.x = np.array(x)