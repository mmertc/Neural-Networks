from lib.model import Model
from lib.Dataset import Dataset
from lib.Dataloader import Dataloader
from lib.Optimizer import Optimizer
from lib.utils import process_epoch, save_checkpoint, load_checkpoint

import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import shutil
import glob




############################################################### 

checkpoints_folder = 'checkpoints'

train_path = 'UCI HAR Dataset/UCI HAR Dataset/train'
train_data_path = os.path.join(train_path, 'X_train.txt')
train_label_path = os.path.join(train_path, 'y_train.txt')

test_path = 'UCI HAR Dataset/UCI HAR Dataset/test'
test_data_path = os.path.join(test_path, 'X_test.txt')
test_label_path = os.path.join(test_path, 'y_test.txt')




############################# - Checkpoint Selection - ################################## 

checkpoint_to_use_folder = 'checkpoint_300_200_05_0001_09_2_01'

#########################################################################################





for file in glob.glob(os.path.join(checkpoints_folder, checkpoint_to_use_folder,"*_best.pkl")):
    checkpoint_to_use = file

checkpoint = load_checkpoint(checkpoint_to_use)
best_val_loss = checkpoint['best_val_loss']
last_epoch = checkpoint['epoch']



epochs = 1


N1= checkpoint['model_parameters']['N1']
N2= checkpoint['model_parameters']['N2']
dropout_rate = checkpoint['model_parameters']['dropout_rate']
learning_rate= checkpoint['optimizer_parameters']['learning_rate']
momentum_const= checkpoint['optimizer_parameters']['momentum_constant']
batch_size= checkpoint['dataloader_parameters']['batch_size']
val_percentage= checkpoint['dataloader_parameters']['val_percentage']


last_epoch = checkpoint['epoch']


###############################################################

dataset = Dataset(
    train_data_path= train_data_path,
    train_label_path= train_label_path,
    test_data_path= test_data_path,
    test_label_path= test_label_path
)


dataloader = Dataloader(
    Dataset= dataset,
    batch_size= None,
    val_percentage= None,
    checkpoint= checkpoint
)


model = Model(
    N1= None,
    N2= None,
    dropout_rate= None,
    checkpoint= checkpoint
)

optimizer = Optimizer(
    learning_rate= None,
    momentum_const= None,
    checkpoint= checkpoint
)



print("---------------------------------------------------------------------------------")
print("Hyperparameter Settings: [Learning Rate: {lr}], [Momentum Constant: {mc}], [N1: {N1}], [N2: {N2}], [Batch-Size: {bs}], [Validation Percentage: {vp}], [Dropout Rate: {dr}]".format(lr=learning_rate, 
                                                                                                                                                                                         mc=momentum_const,
                                                                                                                                                                                         N1=N1,
                                                                                                                                                                                         N2=N2,
                                                                                                                                                                                         bs=batch_size,
                                                                                                                                                                                         vp=val_percentage,
                                                                                                                                                                                         dr=dropout_rate))


process_epoch(
        mode= 'train_test',
        epoch= 1,
        model= model,
        optimizer= optimizer,
        dataloader= dataloader
        ) 

process_epoch(
        mode= 'test',
        epoch= 1,
        model= model,
        optimizer= optimizer,
        dataloader= dataloader
        )                                                                                                                                                                                        