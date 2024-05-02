import os
import numpy as np
from lib.Dataset import Dataset
from sklearn.utils import shuffle
import math


class Dataloader:
    def __init__(self, Dataset, batch_size, val_percentage=0.1, checkpoint=None, mode='train') -> None:


        self.dataset = Dataset
        self.mode = mode
    
        if checkpoint is not None:
            self.batch_size = checkpoint['dataloader_parameters']['batch_size']
            self.val_percentage = checkpoint['dataloader_parameters']['val_percentage']
        else:
            self.batch_size = batch_size
            self.val_percentage = val_percentage


        
        ################################ Train-Val Set
        data = Dataset.train_data
        label = Dataset.train_label.astype(int) - 1

        data_shuffled, label_shuffled = shuffle(data, label)

        train_data = data_shuffled[int(data_shuffled.shape[0]*self.val_percentage):, :]
        val_data = data_shuffled[:int(data_shuffled.shape[0]*self.val_percentage), :]

        train_label = label_shuffled[int(data_shuffled.shape[0]*self.val_percentage):]
        val_label = label_shuffled[:int(data_shuffled.shape[0]*self.val_percentage)]


        N = math.ceil(train_data.shape[0] / self.batch_size)

        train_data_parts = np.array_split(train_data, N, axis=0)
        train_label_parts = np.array_split(train_label, N, axis=0)

        
        self.train_data_whole = train_data
        self.train_label_whole = train_label

        self.train_data_parts = train_data_parts
        self.train_label_parts = train_label_parts

        self.val_data = val_data
        self.val_label = val_label


        ################################ Test Set
        data = Dataset.test_data
        label = Dataset.test_label.astype(int) -1

        self.test_data = data
        self.test_label = label


        self.iterator_max = N
        self.iterator = 0


    def parameters(self):

        dataloader_parameters = {
            'batch_size': self.batch_size,
            'val_percentage': self.val_percentage
        }

        return dataloader_parameters


    def set_mode(self, mode):
        self.mode = mode



    def get_full_batch(self, mode):

        if mode == 'train':

            batch = {
                    "data": self.train_data_whole,
                    "label": self.train_label_whole,
                    "mode": mode,
                    "iterator": 1,
                    "iteration_max": 1
                }
            
        elif mode == 'val':

            batch = {
                    "data": self.val_data,
                    "label": self.val_label,
                    "mode": mode,
                    "iterator": 1,
                    "iteration_max": 1
                }


        elif mode == 'test':

            batch = {
                    "data": self.test_data,
                    "label": self.test_label,
                    "mode": mode,
                    "iterator": 1,
                    "iteration_max": 1
                }
        
        return batch



        
    def __iter__(self):
        return self
    
    def __next__(self):

        if self.iterator == self.iterator_max:
            self.iterator = 0
            raise StopIteration
            

        if self.mode == 'train':    

            batch_train_data = self.train_data_parts[self.iterator]
            batch_train_label = self.train_label_parts[self.iterator]
            batch_val_data =self.val_data
            batch_val_label = self.val_label

            batch = {
                "data": batch_train_data,
                "label": batch_train_label,
                "mode": self.mode,
                "iterator": self.iterator,
                "iteration_max": self.iterator_max
            }

            self.iterator = self.iterator + 1
        
        elif self.mode == 'val':
                
            batch_val_data = self.val_data
            batch_val_label = self.val_label

            batch = {
                "data": batch_val_data,
                "label": batch_val_label,
                "mode": self.mode,
                "iterator": self.iterator,
                "iteration_max": self.iterator_max
            }

        elif self.mode == 'test': 

            batch_test_data = self.test_data
            batch_test_label = self.test_label

            batch = {
                "data": batch_test_data,
                "label": batch_test_label,
                "mode": self.mode,
                "iterator": self.iterator,
                "iteration_max": self.iterator_max
            }

            self.iterator = self.iterator + 1
        
        return batch



