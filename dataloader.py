import pandas as pd

class load_data:
    def __init__(self,path):
        self.file_path = path
    
    def load(self):
        return pd.read_csv(self.file_path)