from lib.model import Model
from lib.Dataset import Dataset
from lib.Dataloader import Dataloader
from lib.Optimizer import Optimizer

import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import shutil

import glob





def process_epoch(
        mode, 
        epoch, 
        model, 
        optimizer, 
        dataloader
    ):

    if mode == 'train':

        ###################################################### Training
        print("---------------------------------------------------------------------------------")

        dataloader.set_mode('train')  
        model.set_mode('train')

        epoch_loss = 0
        for batch in dataloader:
            optimizer.step(model, batch)

            train_batch_loss, train_batch_acc = model.batch_loss_acc(batch)

            
            epoch_loss = epoch_loss + train_batch_loss
        print("Epoch: [{curEpoch}], Step: [{curStep}/{Steps}], Train Loss: {loss:.4f}, Train Acc: {acc:.4f}".format(curEpoch=epoch, 
                                                                                                    curStep=batch['iterator']+1, 
                                                                                                    Steps=batch['iteration_max'], 
                                                                                                    loss=train_batch_loss,
                                                                                                    acc=train_batch_acc))

        ###################################################### Validation
        
        model.set_mode('test')
        val_batch = dataloader.get_full_batch('val')    
        val_batch_loss, val_batch_acc = model.batch_loss_acc(val_batch)

        print("Epoch: [{curEpoch}], Step: [{curStep}/{Steps}], Validation Loss: {loss:.4f}, Validation Acc: {acc:.4f}".format(curEpoch=epoch, 
                                                                                                    curStep=val_batch['iterator'], 
                                                                                                    Steps=val_batch['iteration_max'], 
                                                                                                    loss=val_batch_loss,
                                                                                                    acc=val_batch_acc))
        
        loss_to_return = val_batch_loss

    elif mode == 'train_test':

        model.set_mode('test')
        train_test_batch = dataloader.get_full_batch('train')
        train_test_loss, train_test_batch_acc = model.batch_loss_acc(train_test_batch)
        train_test_match_rates, train_test_mismatch_list = model.rateof_listof_mismatches(train_test_batch)

        print("---------------------------------------------------------------------------------")
        print("Epochs Done: [{curEpoch}], Step: [{curStep}/{Steps}], Train Loss: {loss:.4f}, Train Acc: {acc:.4f}".format(curEpoch=epoch, 
                                                                                                    curStep=1, 
                                                                                                    Steps=1, 
                                                                                                    loss=train_test_loss,
                                                                                                    acc=train_test_batch_acc))
        
        
        print("Training match rates for each pattern: {rate}, \nTraining list of mismatched patterns: {list}".format(rate=train_test_match_rates, 
                                                                                                        list=train_test_mismatch_list))
        print("---------------------------------------------------------------------------------")
        loss_to_return = None
        
    elif mode == 'test':


        model.set_mode('test')
        test_batch = dataloader.get_full_batch('test')
        test_batch_loss, test_batch_acc = model.batch_loss_acc(test_batch)
        test_match_rates, test_mismatch_list = model.rateof_listof_mismatches(test_batch)

        print("---------------------------------------------------------------------------------")
        print("Epochs Done: [{curEpoch}], Step: [{curStep}/{Steps}], Test Loss: {loss:.4f}, Test Acc: {acc:.4f}".format(curEpoch=epoch, 
                                                                                                    curStep=test_batch['iterator'], 
                                                                                                    Steps=test_batch['iteration_max'], 
                                                                                                    loss=test_batch_loss,
                                                                                                    acc=test_batch_acc))
        

        print("Test match rates for each pattern: {rate}, \nTest list of mismatched patterns: {list}".format(rate=test_match_rates, 
                                                                                                        list=test_mismatch_list))
        print("---------------------------------------------------------------------------------")

        loss_to_return = None
        
    return loss_to_return
        



def save_checkpoint(
        model,
        optimizer,
        dataloader,
        checkpoints_folder,
        is_Best,
        best_val_loss,
        epoch
):
    
    state = {
        'model_states': model.states(),
        'optimizer_states': optimizer.states(),
        'model_parameters': model.parameters(),
        'optimizer_parameters': optimizer.parameters(),
        'dataloader_parameters': dataloader.parameters(),
        'best_val_loss': best_val_loss,
        'epoch': epoch
    }

    N1 = model.parameters()['N1']
    N2 = model.parameters()['N2']
    dr = model.parameters()['dropout_rate']
    lr = optimizer.parameters()['learning_rate']
    mc = optimizer.parameters()['momentum_constant']
    bs = dataloader.parameters()['batch_size']
    val_per = dataloader.parameters()['val_percentage']


    cur_checkpoint_basefolder_name = 'checkpoint_{N1}_{N2}_{dr}_{lr}_{mc}_{bs}_{vp}'.format(N1=N1, N2=N2, lr=lr, mc=mc, bs=bs, vp=val_per, dr=dr).replace('.', '')

    cur_checkpoint_file_name = ('epoch_{epoch}_checkpoint_{N1}_{N2}_{dr}_{lr}_{mc}_{bs}_{vp}'.format(N1=N1, N2=N2, lr=lr, mc=mc, bs=bs, vp=val_per, epoch=epoch, dr=dr)).replace('.', '') + '.pkl'
    cur_checkpoint_file_name_best = ('epoch_{epoch}_checkpoint_{N1}_{N2}_{dr}_{lr}_{mc}_{bs}_{vp}_best'.format(N1=N1, N2=N2, lr=lr, mc=mc, bs=bs, vp=val_per, epoch=epoch, dr=dr)).replace('.', '') + '.pkl'

    checkpoint_path = os.path.join(checkpoints_folder, cur_checkpoint_basefolder_name, cur_checkpoint_file_name)
    checkpoint_path_best = os.path.join(checkpoints_folder, cur_checkpoint_basefolder_name, cur_checkpoint_file_name_best)

    cur_checkpoint_dir = dirname(checkpoint_path)

    if cur_checkpoint_dir!='' and not exists(cur_checkpoint_dir):
        makedirs(cur_checkpoint_dir)


    if epoch > 0:
        cur_checkpoint_prev_file_name = ('epoch_{epoch}_checkpoint_{N1}_{N2}_{dr}_{lr}_{mc}_{bs}_{vp}'.format(N1=N1, N2=N2, lr=lr, mc=mc, bs=bs, vp=val_per, epoch=epoch-1, dr=dr)).replace('.', '') + '.pkl'
        try:
            os.remove(os.path.join(dirname(checkpoint_path), cur_checkpoint_prev_file_name))
        except OSError:
            pass


    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state, f)
    if is_Best:
        
        for file in glob.glob(os.path.join(dirname(checkpoint_path),"*_best.pkl")):
            os.remove(file)

        shutil.copyfile(checkpoint_path, checkpoint_path_best)
        




def load_checkpoint(checkpoint_path):

    with open(checkpoint_path, 'rb') as f:
        saved_state = pickle.load(f)
    
    return saved_state