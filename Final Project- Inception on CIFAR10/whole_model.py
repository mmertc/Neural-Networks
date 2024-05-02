import os
from argparse import ArgumentParser

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from data import CIFAR10Data
from module import CIFAR10Module



def main(args):

    seed_everything(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    logger = TensorBoardLogger("cifar10", name="inception")

    checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=False)


    trainer = Trainer(
        fast_dev_run=bool(args.dev),
        logger=logger if not bool(args.dev + args.test_phase) else None,
        gpus=-1,
        deterministic=True,
        weights_summary=None,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        max_epochs=args.max_epochs,
        checkpoint_callback=checkpoint,
        precision=args.precision,
    )

    model = CIFAR10Module(args)
    data = CIFAR10Data(args)

    if bool(args.pretrained):
        state_dict = os.path.join("state_dicts", "inception" + ".pt"
        )
        model.model.load_state_dict(torch.load(state_dict))

    if bool(args.test_phase):
        trainer.test(model, data.test_dataloader())
    else:
        trainer.fit(model, data)
        trainer.test()
    
    model.save_metrics()

        


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="data/cifar10")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu_id", type=str, default="0")

    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--aux_branches", type=bool, default=False)
    parser.add_argument("--label_smoothing", type=bool, default=False)

    args = parser.parse_args()
    main(args)



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



import math
import warnings
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        last_epoch: int = -1,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]



import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
from torchmetrics import Accuracy
import torch


def load_metrics():

    with open('metrics/saved_metrics/lbl_v1.pkl', 'rb') as f:
        loaded_metrics = pickle.load(f)

    return  loaded_metrics['train_metrics'], loaded_metrics['val_metrics'], loaded_metrics['train_outs'], loaded_metrics['val_outs'], loaded_metrics['test_outs']



train_metrics, val_metrics, train_outs, val_outs, test_outs = load_metrics()





fig, axs = plt.subplots(1, 3)

# Loss per epoch

train_losses = []
val_losses = []
index = []
for i in range(len(train_metrics)):
    loss_train = train_metrics[i]['loss'].cpu().detach().numpy().item()
    loss_val = val_metrics[i]['loss'].cpu().detach().numpy().item()

    index.append(i)
    train_losses.append(loss_train)
    val_losses.append(loss_val)


axs[0].plot(index, train_losses)
axs[0].plot(index, val_losses)
axs[0].set(xlabel='Epoch', ylabel='Average Loss')
axs[0].set_title('Average Loss vs Epoch')
axs[0].legend(['Train', 'Validation'])


#################


# Train Accuracy per Epoch

train_accuracy1 = []
train_accuracy2 = []
train_accuracy3 = []
index = []
for i in range(len(train_metrics)):
    acc1 = train_metrics[i]['acc1'].cpu().detach().numpy().item()
    acc2 = train_metrics[i]['acc2'].cpu().detach().numpy().item()
    acc3 = train_metrics[i]['acc3'].cpu().detach().numpy().item()

    index.append(i)
    train_accuracy1.append(acc1)
    train_accuracy2.append(acc2)
    train_accuracy3.append(acc3)
    
axs[1].plot(index, train_accuracy1)
axs[1].plot(index, train_accuracy2)
axs[1].plot(index, train_accuracy3)
axs[1].set(xlabel='Epoch', ylabel='Accuracy')
axs[1].set_title('Train Accuracy vs Epoch')
axs[1].legend(['Top-1 Accuracy', 'Top-2 Accuracy', 'Top-3 Accuracy'])


#################


# Val Accuracy per Epoch

val_accuracy1 = []
val_accuracy2 = []
val_accuracy3 = []
index = []
for i in range(len(val_metrics)):
    acc1 = val_metrics[i]['acc1'].cpu().detach().numpy().item()
    acc2 = val_metrics[i]['acc2'].cpu().detach().numpy().item()
    acc3 = val_metrics[i]['acc3'].cpu().detach().numpy().item()

    index.append(i)
    val_accuracy1.append(acc1)
    val_accuracy2.append(acc2)
    val_accuracy3.append(acc3)
    
axs[2].plot(index, val_accuracy1)
axs[2].plot(index, val_accuracy2)
axs[2].plot(index, val_accuracy3)
axs[2].set(xlabel='Epoch', ylabel='Accuracy')
axs[2].set_title('Validation Accuracy vs Epoch')
axs[2].legend(['Top-1 Accuracy', 'Top-2 Accuracy', 'Top-3 Accuracy'])
fig.suptitle("Loss and Accuracy Metrics for the model")
plt.show()


#################

fig, axs = plt.subplots(1, 3)


# Train Confusion Matrix 

cm = confusion_matrix(train_outs['labels'], train_outs['preds'])

disp = ConfusionMatrixDisplay(cm)
disp.plot(ax=axs[0])
disp.ax_.set_title('Train Confusion Matrix')


#################



# Val Confusion Matrix 

cm = confusion_matrix(val_outs['labels'], val_outs['preds'])

disp = ConfusionMatrixDisplay(cm)
disp.plot(ax=axs[1])
disp.ax_.set_title('Validation Confusion Matrix')


#################



# Test Confusion Matrix 

cm = confusion_matrix(test_outs['labels'], test_outs['preds'])

disp = ConfusionMatrixDisplay(cm)
disp.plot(ax=axs[2])
disp.ax_.set_title('Test Confusion Matrix')
fig.suptitle("Confusion Metrics for the model")
plt.show()

#################


# Train Accuracy

measureAcc1 = Accuracy(top_k=1)
measureAcc2 = Accuracy(top_k=2)
measureAcc3 = Accuracy(top_k=3)

train_acc1 = measureAcc1(torch.from_numpy(train_outs['pred_probs']), torch.from_numpy(train_outs['labels']).int())
train_acc2 = measureAcc2(torch.from_numpy(train_outs['pred_probs']), torch.from_numpy(train_outs['labels']).int())
train_acc3 = measureAcc3(torch.from_numpy(train_outs['pred_probs']), torch.from_numpy(train_outs['labels']).int())

print("[Train set] Top-1 Accuracy: {acc1:.4f}, Top-2 Accuracy: {acc2:.4f}, Top-3 Accuracy: {acc3:.4f},".format(acc1=train_acc1*100, acc2=train_acc2*100, acc3=train_acc3*100))

#################


# Val Accuracy


val_acc1 = measureAcc1(torch.from_numpy(val_outs['pred_probs']), torch.from_numpy(val_outs['labels']).int())
val_acc2 = measureAcc2(torch.from_numpy(val_outs['pred_probs']), torch.from_numpy(val_outs['labels']).int())
val_acc3 = measureAcc3(torch.from_numpy(val_outs['pred_probs']), torch.from_numpy(val_outs['labels']).int())

print("[Validation set] Top-1 Accuracy: {acc1:.4f}, Top-2 Accuracy: {acc2:.4f}, Top-3 Accuracy: {acc3:.4f},".format(acc1=val_acc1*100, acc2=val_acc2*100, acc3=val_acc3*100))

#################


# Test Accuracy


test_acc1 = measureAcc1(torch.from_numpy(test_outs['pred_probs']), torch.from_numpy(test_outs['labels']).int())
test_acc2 = measureAcc2(torch.from_numpy(test_outs['pred_probs']), torch.from_numpy(test_outs['labels']).int())
test_acc3 = measureAcc3(torch.from_numpy(test_outs['pred_probs']), torch.from_numpy(test_outs['labels']).int())

print("[Test set] Top-1 Accuracy: {acc1:.4f}, Top-2 Accuracy: {acc2:.4f}, Top-3 Accuracy: {acc3:.4f},".format(acc1=test_acc1*100, acc2=test_acc2*100, acc3=test_acc3*100))

#################


import os
import zipfile

import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)


    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform)
        #subset = Subset(dataset, range(256))

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform)
        #subset = Subset(dataset, range(256))

        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _WeightedLoss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(-1)

        with torch.no_grad():
            dist = torch.zeros_like(pred)
            dist.fill_(self.smoothing / (self.cls - 1))
            dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-dist * pred, dim=self.dim))