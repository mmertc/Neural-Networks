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





############################################################### 

checkpoints_folder = 'checkpoints'

train_path = 'UCI HAR Dataset/UCI HAR Dataset/train'
train_data_path = os.path.join(train_path, 'X_train.txt')
train_label_path = os.path.join(train_path, 'y_train.txt')

test_path = 'UCI HAR Dataset/UCI HAR Dataset/test'
test_data_path = os.path.join(test_path, 'X_test.txt')
test_label_path = os.path.join(test_path, 'y_test.txt')


############################################################### 


#checkpoint_to_use = 'checkpoint_300_200_001_09_50_01/epoch_39_checkpoint_300_200_001_09_50_01.pkl'
checkpoint_to_use = ''


epochs = 100


N1= 300
N2= 200
learning_rate= 0.0001
momentum_const= 0.9
batch_size= 2
val_percentage= 0.1
dropout_rate = 0.8

checkpoint = None

best_val_loss = float('Inf')
last_epoch = 0

if checkpoint_to_use != '':
    checkpoint = load_checkpoint(os.path.join(checkpoints_folder, checkpoint_to_use))
    best_val_loss = checkpoint['best_val_loss']
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
    batch_size= batch_size,
    val_percentage= val_percentage,
    checkpoint= checkpoint
)


model = Model(
    N1= N1,
    N2= N2,
    dropout_rate= dropout_rate,
    checkpoint= checkpoint
)

optimizer = Optimizer(
    learning_rate= learning_rate,
    momentum_const= momentum_const,
    checkpoint= checkpoint
)




val_loss_history = np.zeros(last_epoch+epochs+2)
val_loss_history[0] = best_val_loss

print("---------------------------------------------------------------------------------")
print("Hyperparameter Settings: [Learning Rate: {lr}], [Momentum Constant: {mc}], [N1: {N1}], [N2: {N2}], [Batch-Size: {bs}], [Validation Percentage: {vp}], [Dropout Rate: {dr}]".format(lr=learning_rate, 
                                                                                                                                                                                         mc=momentum_const,
                                                                                                                                                                                         N1=N1,
                                                                                                                                                                                         N2=N2,
                                                                                                                                                                                         bs=batch_size,
                                                                                                                                                                                         vp=val_percentage,
                                                                                                                                                                                         dr=dropout_rate))




for epoch_id in range(last_epoch+1, last_epoch+epochs+1):


    ########################### Perform an Epoch
    val_loss_history[epoch_id-last_epoch+1] = process_epoch(
        mode= 'train',
        epoch= epoch_id,
        model= model,
        optimizer= optimizer,
        dataloader= dataloader
        )
    
    if val_loss_history[epoch_id-last_epoch+1] < best_val_loss:
        best_val_loss = val_loss_history[epoch_id-last_epoch+1]
        is_Best = True
    else: 
        is_Best = False


    
    ########################### Save the State
    save_checkpoint(
    model=model,
    optimizer= optimizer,
    dataloader= dataloader,
    checkpoints_folder= checkpoints_folder,
    is_Best= is_Best,
    best_val_loss= best_val_loss,
    epoch= epoch_id
    )




process_epoch(
        mode= 'test',
        epoch= epoch_id,
        model= model,
        optimizer= optimizer,
        dataloader= dataloader
        )













        

    



    
    




           



    










