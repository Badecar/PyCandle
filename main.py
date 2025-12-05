#%%
from utils.dataloader import DataLoader, train_dataset, test_dataset
from utils.initializer import NormalInitializer
from utils.loss_function import cross_entropy_loss
from utils.cn import *
from utils.tensor import Tensor
from utils import candle
from training.traning import train_model, eval_model, get_acc
import numpy as np
from utils.optimizer import SGD, SGDMomentum, ADAM
import matplotlib.pyplot as plt
import wandb

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 10
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

        # self.layers = Sicquential(
        #     Flatten(),
        #     Linear(n_in=in_channels, n_out=600, bias=True),
        #     ReLU(),
        #     Linear(n_in=600, n_out= 600, bias=True),
        #     ReLU(),
        #     Linear(n_in=600, n_out=120, bias=True),
        #     ReLU(),
        #     Linear(n_in=120, n_out=num_classes, bias=True)
        # )

        self.layers = Sicquential(
            Conv2D(in_channels=1, num_kernels=32, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            Conv2D(in_channels=32, num_kernels=32, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            MaxPool2D(2,2), #14x14

            Conv2D(in_channels=32, num_kernels=64, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            Conv2D(in_channels=64, num_kernels=64, kernel_size=3, stride=1, padding="same"),
            ReLU(),
            MaxPool2D(2,2), #7x7

            Flatten(),
            Linear(n_in=64*7*7, n_out=512, bias=True),
            ReLU(),
            Linear(n_in=512, n_out=num_classes, bias=True)
        )
    
    def forward(self, x):
        return self.layers(x)
        

model = Model(in_channels=28*28, num_classes=10)
optimizer = ADAM(model.parameters(), lr=LEARNING_RATE)
model, loss_list = train_model(model, train_loader, test_loader, optimizer, cross_entropy_loss, NUM_EPOCHS, use_wandb=use_wandb)

metrics = eval_model(model, test_loader, plot_confusion_matrix=False)
print("f1 =", metrics['f1_mean'])

plt.plot(loss_list)
plt.show()

if use_wandb:
    run.finish()

