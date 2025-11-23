#%%
from utils.dataloader import DataLoader, train_dataset
from utils.initializer import NormalInitializer
from utils.loss_function import cross_entropy_loss
from utils.cn import *
from utils.tensor import Tensor
import numpy as np
from utils.optimizer import SGD
import matplotlib.pyplot as plt

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

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

optimizer = SGD(model.parameters(), lr=0.003)

loss_list = []
for i, batch in enumerate(train_loader):
  x, y = batch

  output = model(x)
  loss = cross_entropy_loss(y, output)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  print(f"Batch {i}, Loss: {loss.v:.4f}")
  loss_list.append(loss.v)


  if i >= 1000:  # Run 20 batches to see convergence
    break

plt.plot(loss_list)
plt.show()

# %%
