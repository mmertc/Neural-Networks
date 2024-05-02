import os
import numpy as np


class Dataset:
    def __init__(self, train_data_path , train_label_path, test_data_path, test_label_path) -> None:
        
        with open(train_data_path) as f:
            lines = f.readlines()

        data = []
        for line in lines:
            data.append(line.split())

        self.train_data = np.array(data, dtype=np.float32)



        with open(train_label_path) as f:
            lines = f.readlines()

        data = []
        for line in lines:
            data.append(line.split())

        self.train_label = np.array(data, dtype=np.float32)



        with open(test_data_path) as f:
            lines = f.readlines()

        data = []
        for line in lines:
            data.append(line.split())

        self.test_data = np.array(data, dtype=np.float32)



        with open(test_label_path) as f:
            lines = f.readlines()

        data = []
        for line in lines:
            data.append(line.split())

        self.test_label = np.array(data, dtype=np.float32)