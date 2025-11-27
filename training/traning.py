import wandb
import numpy as np
from utils import candle
from utils.cn import Module
from utils.dataloader import DataLoader

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
            
            # if i >= 800:
            #     break
    
    return model, loss_list

def eval_model(model:Module, loader:DataLoader):
    model.eval()
    total_acc = 0

    with candle.no_grad():
        for input_batch, label_batch in loader:
            label_batch = np.argmax(label_batch.v, axis=1)
            logits = model(input_batch).v
            classifications = np.argmax(logits, axis=1)
            total_acc += (classifications == label_batch).sum().item()
    
    total_acc = total_acc / len(loader.dataset)
    return total_acc