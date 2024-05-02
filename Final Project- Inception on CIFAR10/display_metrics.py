
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