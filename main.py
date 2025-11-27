#%%
from utils.dataloader import DataLoader, train_dataset, test_dataset
from utils.initializer import NormalInitializer
from utils.loss_function import cross_entropy_loss
from utils.cn import *
from utils.tensor import Tensor
from utils import candle
from training.traning import train_model, eval_model
import numpy as np
from utils.optimizer import SGD, SGDMomentum, ADAM
import matplotlib.pyplot as plt
import wandb

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 1
use_wandb = False

# Configuring wandb
if use_wandb:
    run = wandb.init(project="simple-nn-from-scratch",
                #  entity="s234817",
                 name="simple-nn" ,
                 reinit=True )
    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS
    }

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

class Model(Module):
    def __init__(self, num_classes:int, in_channels:int):
        super().__init__()

        self.layers = Sicquential(
            Linear(n_in=in_channels, n_out=600),
            ReLU(),
            Linear(n_in=600, n_out= 600),
            ReLU(),
            Linear(n_in=600, n_out=120),
            ReLU(),
            Linear(n_in=120, n_out=num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

model = Model(in_channels=28*28, num_classes=10)
optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

model, loss_list = train_model(model, train_loader, optimizer, cross_entropy_loss, NUM_EPOCHS, use_wandb=use_wandb)

acc = eval_model(model, test_loader)
print("acc=", acc)

plt.plot(loss_list)
plt.show()

if use_wandb:
    run.finish()

