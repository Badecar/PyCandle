import wandb
import numpy as np
from utils import candle
from utils.cn import Module
from utils.dataloader import DataLoader
from utils.loss_function import cross_entropy_loss
from plotting.plotting import plot_confusion_matrix_f
from utils.tensor import Tensor
import seaborn as sns
import matplotlib.pyplot as plt

def train_model(model:Module, loader, val_loader, optim, criterion, epochs, use_wandb=False):
    loss_list = []
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(loader):
            x, y = batch

            output = model(x)
            loss = criterion(y, output)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Calculate training accuracy
            y_true = np.argmax(y.v, axis=1)
            y_pred = np.argmax(output.v, axis=1)
            acc = (y_true == y_pred).sum() / len(y_true)

            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.v:.4f}, Acc: {acc:.4f}")
            loss_list.append(loss.v)

            if use_wandb:
                wandb.log({"train/loss": loss.v, "train/acc": acc})
        
        # Validation step
        if val_loader is not None:
            val_metrics = eval_model(model, val_loader, plot_confusion_matrix=False)
            print(f"Epoch {epoch} Validation: Loss: {val_metrics['cross_entropy']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            if use_wandb:
                wandb.log({
                    "val/loss": val_metrics['cross_entropy'],
                    "val/acc": val_metrics['accuracy']
                })
    
    return model, loss_list


def eval_model(model:Module, loader:DataLoader , plot_confusion_matrix = False):

    """
    """
    model.eval()
    metrics = {
        'accuracy': 0,
        'f1_classes': [],
        'f1_mean': 0,
        'cross_entropy': 0,
        'confusion_matrix': []      
    }
    
    # Inference
    y_pred = []
    y = []
    with candle.no_grad():
        for X, Y in loader:
            logits = model(X)
            y_batch = np.argmax(logits.v, axis=1)
            y_pred_batch = np.argmax(Y.v, axis=1)

            metrics['cross_entropy'] += float(cross_entropy_loss(Y, logits).v) * 1/len(loader)

            y_pred.extend(y_pred_batch)
            y.extend(y_batch)

    y_pred = np.array(y_pred)
    y = np.array(y)

    metrics['accuracy'] = float((y == y_pred).sum() / len(y)) # TP / # Datapoints

    # Confusion matrix
    num_classes = loader.dataset.y.shape[1]  # one-hot, so 2nd dim = num classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y, y_pred):
        confusion_matrix[t, p] += 1

    # F1 scores (class based)
    f1_scores = []
    for cls in range(num_classes):
        TP = confusion_matrix[cls, cls]
        FP = confusion_matrix[:, cls].sum() - TP
        FN = confusion_matrix[cls, :].sum() - TP
        TN = len(y) - (TP + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)

    metrics['f1_classes'] = f1_scores
    metrics['f1_mean'] = float(np.mean(f1_scores))
    metrics['confusion_matrix'] = confusion_matrix

    if plot_confusion_matrix:
        plot_confusion_matrix_f(metrics['confusion_matrix'], loader, num_classes=10)

    return metrics


def get_acc(model:Module, loader:DataLoader):
    model.eval()
    total_acc = 0
    dataset_size = len(loader)*loader.batch_size if loader.drop_last else len(loader.dataset)

    with candle.no_grad():
        for X, Y in loader:
            Y = np.argmax(Y.v, axis=1)
            logits = model(X).v
            y_pred = np.argmax(logits, axis=1)
            total_acc += (y_pred == Y).sum()
    
    total_acc = total_acc / dataset_size
    return total_acc