from utils import cn

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