#%%
from utils.dataloader import DataLoader, train_dataset
from utils.loss_function import cross_entropy_loss
from utils.cn import *
from utils.optimizer import SGD, SGDMomentum, ADAM
import matplotlib.pyplot as plt
import wandb

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.01
BATCH_SIZE = 128
NUM_EPOCHS = 5
weights_and_biases = False

# Configuring wandb
if weights_and_biases:
    # Integrating wandb with the experiment
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

class Model(Module):
    def __init__(self, num_classes:int, in_channels:int):
        super().__init__()

        self.layers = Sicquential(
            Linear(n_in=in_channels, n_out=600, bias=True),
            ReLU(),
            Linear(n_in=600, n_out= 600, bias=True),
            ReLU(),
            Linear(n_in=600, n_out=120, bias=True),
            ReLU(),
            Linear(n_in=120, n_out=num_classes, bias=True)
        )
    
    def forward(self, x):
        return self.layers(x)

model = Model(in_channels=28*28, num_classes=10)

optimizer = ADAM(model.parameters(), lr=LEARNING_RATE)

loss_list = []
for epoch in range(NUM_EPOCHS):

    for i, batch in enumerate(train_loader):
        x, y = batch

        output = model(x)
        loss = cross_entropy_loss(y, output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Batch {i}, Loss: {loss.v:.4f}")
        loss_list.append(loss.v)
        if weights_and_biases:
            wandb.log({"train/loss": loss.v})

plt.plot(loss_list)
plt.show()

if weights_and_biases:
    run.finish()

