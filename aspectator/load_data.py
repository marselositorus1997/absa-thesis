  
'''Contains classes to load data'''

import os
import pandas as pd
import numpy as np
class DataLoader(object):

    def __init__(self, base_folder):
        self.base_folder = base_folder

    def load_data(self, file_path):
        path = os.path.join(self.base_folder,file_path, sep = ';')
        df = pd.read_csv(path)
        #find random sample
        np.random.seed(2024)
        idx = np.random.randint(low=0, high=len(df), size = 2)
        sample_df = df.iloc[idx]
        return sample_df

    def load_data2(self, file_path):
        path = os.path.join(self.base_folder,file_path)
        df = pd.read_csv(path)
        #find random sample
        np.random.seed(2024)
        idx = np.random.randint(low=0, high=len(df), size = 2)
        sample_df = df.iloc[idx]
        return sample_df
