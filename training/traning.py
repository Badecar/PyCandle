from typing import Any


import wandb
import numpy as np
from utils import candle
from utils.cn import Module
from utils.dataloader import DataLoader
from utils.loss_function import cross_entropy_loss
from utils.tensor import Tensor
import seaborn as sns
import matplotlib.pyplot as plt

def train_model(model:Module, loader, optim, criterion, epochs, use_wandb):
    model.train()
    loss_list = []
    for epoch in range(epochs):

        for i, batch in enumerate(loader):
            x, y = batch

            output = model(x)
            loss = criterion(y, output)
            loss.backward()
            optim.step()
            optim.zero_grad()
            print(f"Batch {i}, Loss: {loss.v:.4f}")
            loss_list.append(loss.v)
            if use_wandb:
                wandb.log({"train/loss": loss.v})
            
            if i >= 200:
                break
    
    return model, loss_list

def eval_model(model:Module, loader:DataLoader , plot_confusion_matrix = False):

    """
    """
    model.eval()
    # 1. Defining the metrics that we would like to compute
    metrics = {
        'accuracy': 0,
        'f1 score': 0,
        'cross entropy': 0        
    }
    # accuracy_per_class = np.zeros(10) # If oscar wants to implement.

    # 2. Inference
    # Get the predictions
    X = Tensor(loader.dataset.X)
    y_pred = np.argmax(model(X).v , axis=1)
    # Load y
    y = np.argmax(loader.dataset.y , axis = 1)
    
    # 3. Metrics
    # Accuracy
    metrics['accuracy'] = (y == y_pred).sum() / len(y) # TP / # Datapoints

    # CE
    # To compute cross-entropy, we need one-hot encoded y and logits from the model
    logits = model(X)                    # model output logits
    y_onehot = Tensor(loader.dataset.y)  # y is already one-hot shape (batch, classes)
    metrics['cross entropy'] = float(cross_entropy_loss(y_onehot, logits).v)
    
    #  3.5 Before F1, we will make the confusion matrice ( we need True positives, False Positives etc. to make f1)
    # Build confusion matrix
    num_classes = loader.dataset.y.shape[1]  # one-hot, so 2nd dim = num classes
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y, y_pred):
        confusion_matrix[t, p] += 1

    # 3.5 Continued: F1 score (macro average). 
    f1_scores = []
    for cls in range(num_classes):
        TP = confusion_matrix[cls, cls]
        FP = confusion_matrix[:, cls].sum() - TP
        FN = confusion_matrix[cls, :].sum() - TP
        TN = len(y) - (TP + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    metrics['f1 score'] = float(np.mean(f1_scores))
    metrics['confusion_matrix'] = confusion_matrix

    # 4. Plotting if necesary
    if plot_confusion_matrix:
        fig, ax = plt.subplots(figsize=(8, 6))

        class_labels = getattr(loader.dataset, 'class_names', None)
        if class_labels is None:
            class_labels = [str(i) for i in range(num_classes)]

        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax
        )

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.show()


    return metrics