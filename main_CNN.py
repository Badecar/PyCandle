#%%
from utils.dataloader import DataLoader, train_dataset, test_dataset
from utils.initializer import NormalInitializer
from utils.loss_function import cross_entropy_loss
from utils import cn
from training.traning import train_model, eval_model, get_acc
from utils import optimizer
import matplotlib.pyplot as plt

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 1

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

class CNN(cn.Module):
    def __init__(self, num_classes:int, in_channels:int):
        super().__init__()

        self.layers = cn.Sequential(
            cn.Conv2D(in_channels=1, num_kernels=32, kernel_size=3, stride=1, padding="same"),
            cn.ReLU(),
            cn.Conv2D(in_channels=32, num_kernels=32, kernel_size=3, stride=1, padding="same"),
            cn.ReLU(),
            cn.MaxPool2D(2,2), #14x14

            cn.Conv2D(in_channels=32, num_kernels=64, kernel_size=3, stride=1, padding="same"),
            cn.ReLU(),
            cn.Conv2D(in_channels=64, num_kernels=64, kernel_size=3, stride=1, padding="same"),
            cn.ReLU(),
            cn.MaxPool2D(2,2), #7x7

            cn.Flatten(),
            cn.Linear(n_in=64*7*7, n_out=512, bias=True),
            cn.ReLU(),
            cn.Linear(n_in=512, n_out=num_classes, bias=True)
        )

    def forward(self, x):
        return self.layers(x)

model = CNN(in_channels=28*28, num_classes=10)
optimizer = optimizer.ADAM(model.parameters(), lr=LEARNING_RATE)
model, loss_list = train_model(model, train_loader, test_loader, optimizer, cross_entropy_loss, NUM_EPOCHS)

metrics = eval_model(model, test_loader, plot_confusion_matrix=False)
print("f1 =", metrics['f1_mean'])

plt.plot(loss_list)
plt.show()
