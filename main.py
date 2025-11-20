#%%
from utils.dataloader import DataLoader, train_dataset
from utils.initializer import NormalInitializer
from utils.loss_function import cross_entropy_loss
from utils.nn import Module
from utils.tensor import Tensor
import numpy as np
from utils.optimizer import SGD
import matplotlib.pyplot as plt

class SingleLayer(Module):
    def __init__(self):
        super().__init__()
        self.initializer = NormalInitializer()
        self.parameter = self.initializer.init_weights(784, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.parameter
        return x


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
model = SingleLayer()
optimizer = SGD(model.parameters(), lr=0.01)  # Now stable with proper gradients!

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
