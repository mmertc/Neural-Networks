from typing import Any, List
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

import numpy as np
import pickle 


from label_smoothing_loss import LabelSmoothingLoss
from model.inception import InceptionModel

from schduler import WarmupCosineLR



class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.label_smoothing = self.hparams.label_smoothing
        self.aux_branches = self.hparams.aux_branches

        #Loss function selection, with label smoothing or without.
        if self.label_smoothing:
            self.criterion = LabelSmoothingLoss(10)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        

        self.accuracy = Accuracy()
        self.accuracy2 = Accuracy(top_k=2)
        self.accuracy3 = Accuracy(top_k=3)


        self.model = InceptionModel(aux_logits=self.aux_branches)
        

        self.epoch_metrics_train = []
        self.epoch_metrics_val = []

        self.cur_epoch_train_metrics = []
        self.cur_epoch_val_metrics = []

        self.cur_train_preds = []
        self.cur_val_preds = []
        self.cur_test_preds = []

        self.train_preds = []
        self.val_preds = []
        self.test_preds = []



    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)

        #This is to handle multiple outputs if auxiliary branches exist.
        if self.training and self.aux_branches:
            loss_main = self.criterion(predictions[0], labels)
            loss_aux_2 = self.criterion(predictions[1], labels)
            loss_aux_1 = self.criterion(predictions[2], labels)



            accuracy = self.accuracy(predictions[0], labels)
            accuracy2 = self.accuracy2(predictions[0], labels)
            accuracy3 = self.accuracy3(predictions[0], labels)

            predictions = predictions[0]

        else:
            loss_main = self.criterion(predictions, labels)
            loss_aux_2 = 0
            loss_aux_1 = 0

            accuracy = self.accuracy(predictions, labels)
            accuracy2 = self.accuracy2(predictions, labels)
            accuracy3 = self.accuracy3(predictions, labels)

            predictions = predictions


        #The losses from the auxiliary branches are added with discount weights, the values are 0 if they do not exist.
        loss = loss_main + 0.3 * loss_aux_2 + 0.3 * loss_aux_1

        return loss, accuracy * 100, accuracy2 * 100, accuracy3 * 100, predictions
    


    def on_train_epoch_start(self):

        self.cur_train_preds = []
        self.cur_val_preds = []
        self.cur_test_preds = []


    def training_step(self, batch, batch_nb):
        loss, accuracy, acc2, acc3, preds = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)

        metrics = {
                'loss':loss,
                'acc1':accuracy,
                'acc2':acc2,
                'acc3':acc3
            }

        self.cur_epoch_train_metrics.append(metrics)


        pred_probs = preds.cpu().detach().numpy()
        preds = np.argmax(pred_probs, axis=1)

        outs = {
            'preds':preds,
            'pred_probs':pred_probs,
            'labels':batch[1]
        }

        self.cur_train_preds.append(outs)


        return loss
    


    def validation_step(self, batch, batch_nb):
        loss, accuracy, acc2, acc3, preds = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)
        

        metrics = {
                'loss':loss,
                'acc1':accuracy,
                'acc2':acc2,
                'acc3':acc3
            }

        self.cur_epoch_val_metrics.append(metrics)



        pred_probs = preds.cpu().detach().numpy()
        preds = np.argmax(pred_probs, axis=1)

        outs = {
            'preds':preds,
            'pred_probs':pred_probs,
            'labels':batch[1]
        }

        self.cur_val_preds.append(outs)


    def on_train_epoch_end(self, outputs):

        epoch_loss = 0
        acc1 = 0
        acc2 = 0
        acc3 = 0
        for metric in self.cur_epoch_train_metrics:
            epoch_loss += metric['loss']
            acc1 += metric['acc1']
            acc2 += metric['acc2']
            acc3 += metric['acc3']
        epoch_loss /= len(self.cur_epoch_train_metrics)
        acc1 /= len(self.cur_epoch_train_metrics)
        acc2 /= len(self.cur_epoch_train_metrics)
        acc3 /= len(self.cur_epoch_train_metrics)

        epoch_metrics = {
            'loss':epoch_loss,
            'acc1':acc1,
            'acc2':acc2,
            'acc3':acc3
        }

        self.cur_epoch_train_metrics = []
        self.epoch_metrics_train.append(epoch_metrics)


        preds_all = np.empty(1)
        pred_probs_all = np.empty((1, 10))
        labels_all = np.empty(1)
        for out in self.cur_train_preds:
            preds = out['preds']
            pred_probs = out['pred_probs']
            labels = out['labels'].cpu().detach().numpy()

            preds_all = np.concatenate((preds_all, preds))
            pred_probs_all = np.concatenate((pred_probs_all, pred_probs))
            labels_all = np.concatenate((labels_all, labels))


        preds_all = preds_all[1:]
        pred_probs_all = pred_probs_all[1:,:]
        labels_all = labels_all[1:]


        out = {
            'preds':preds_all,
            'pred_probs':pred_probs_all,
            'labels':labels_all
        }

        self.train_preds = out




    
    def on_validation_epoch_end(self):

        epoch_loss = 0
        acc1 = 0
        acc2 = 0
        acc3 = 0
        for metric in self.cur_epoch_val_metrics:
            epoch_loss += metric['loss']
            acc1 += metric['acc1']
            acc2 += metric['acc2']
            acc3 += metric['acc3']
        epoch_loss /= len(self.cur_epoch_val_metrics)
        acc1 /= len(self.cur_epoch_val_metrics)
        acc2 /= len(self.cur_epoch_val_metrics)
        acc3 /= len(self.cur_epoch_val_metrics)

        epoch_metrics = {
            'loss':epoch_loss,
            'acc1':acc1,
            'acc2':acc2,
            'acc3':acc3
        }

        self.cur_epoch_val_metrics = []
        self.epoch_metrics_val.append(epoch_metrics)




        preds_all = np.empty(1)
        pred_probs_all = np.empty((1, 10))
        labels_all = np.empty(1)
        for out in self.cur_val_preds:
            preds = out['preds']
            pred_probs = out['pred_probs']
            labels = out['labels'].cpu().detach().numpy()

            preds_all = np.concatenate((preds_all, preds))
            pred_probs_all = np.concatenate((pred_probs_all, pred_probs))
            labels_all = np.concatenate((labels_all, labels))


        preds_all = preds_all[1:]
        pred_probs_all = pred_probs_all[1:,:]
        labels_all = labels_all[1:]


        out = {
            'preds':preds_all,
            'pred_probs':pred_probs_all,
            'labels':labels_all
        }

        self.val_preds = out




    def on_test_epoch_end(self):
        
        preds_all = np.empty(1)
        pred_probs_all = np.empty((1, 10))
        labels_all = np.empty(1)
        for out in self.cur_test_preds:
            preds = out['preds']
            pred_probs = out['pred_probs']
            labels = out['labels'].cpu().detach().numpy()

            preds_all = np.concatenate((preds_all, preds))
            pred_probs_all = np.concatenate((pred_probs_all, pred_probs))
            labels_all = np.concatenate((labels_all, labels))


        preds_all = preds_all[1:]
        pred_probs_all = pred_probs_all[1:,:]
        labels_all = labels_all[1:]


        out = {
            'preds':preds_all,
            'pred_probs':pred_probs_all,
            'labels':labels_all
        }

        self.test_preds = out



    def get_metrics(self):

        train_metrics = self.epoch_metrics_train
        val_metrics = self.epoch_metrics_val[1:]

        train_outs = self.train_preds
        val_outs = self.val_preds
        test_outs = self.test_preds


        return train_metrics, val_metrics, train_outs, val_outs, test_outs
    

    def save_metrics(self):

        train_metrics, val_metrics, train_outs, val_outs, test_outs = self.get_metrics()

        save = {
            'train_metrics':train_metrics,
            'val_metrics':val_metrics,
            'train_outs':train_outs,
            'val_outs':val_outs,
            'test_outs':test_outs
        }

        with open('metrics/saved_metrics.pkl', 'wb') as f:
            pickle.dump(save, f)

    
    def load_metrics():

        with open('metrics/saved_metrics.pkl', 'rb') as f:
            loaded_metrics = pickle.load(f)

        return loaded_metrics['train_metrics'], loaded_metrics['val_metrics'], loaded_metrics['train_outs'], loaded_metrics['val_outs'], loaded_metrics['test_outs']



    def test_step(self, batch, batch_nb):
        loss, accuracy, acc2, acc3, preds = self.forward(batch)
        self.log("acc/test", accuracy)



        pred_probs = preds.cpu().detach().numpy()
        preds = np.argmax(pred_probs, axis=1)

        outs = {
            'preds':preds,
            'pred_probs':pred_probs,
            'labels':batch[1]
        }

        self.cur_test_preds.append(outs)


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
