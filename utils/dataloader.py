#%%
import kagglehub
import numpy as np
import pandas as pd
from utils.tensor import Tensor
# Download latest version
path = kagglehub.dataset_download("zalando-research/fashionmnist")

# print("Path to dataset files:", path)

train_df = pd.read_csv(path + "/fashion-mnist_train.csv")
test_df  = pd.read_csv(path + "/fashion-mnist_test.csv")

# print("Train data shape:", train_df.shape)
# print("Test data shape:", test_df.shape)

class_names = [
    "T-shirt/top",  
    "Trouser",     
    "Pullover",   
    "Dress",       
    "Coat",       
    "Sandal",    
    "Shirt",      
    "Sneaker",     
    "Bag",        
    "Ankle boot"   
]

X_train_full = train_df.drop("label", axis=1).values / 255.0  # Normalize to [0, 1]
y_train_full = np.eye(len(class_names))[train_df["label"].values]

X_test = test_df.drop("label", axis=1).values / 255.0  # Normalize to [0, 1]
y_test = np.eye(len(class_names))[test_df["label"].values]
#%%
from abc import ABC, abstractmethod

class Dataset(ABC):
  @abstractmethod
  def __init__(self, X, y):
    pass

  @abstractmethod
  def __len__(self):
    pass

  @abstractmethod
  def __getitem__(self, idx):
    pass

class FashionMNISTDataset(Dataset):
  def __init__(self, X, y):
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return Tensor(self.X[idx]), Tensor(self.y[idx])

  def __repr__(self):
    return f"FashionMNISTDataset(X={self.X.shape}, y={self.y.shape})"


train_dataset = FashionMNISTDataset(X_train_full, y_train_full)
#%%

class DataLoader():
  def __init__(self, dataset: Dataset, batch_size = 1, shuffle=False, drop_last=False):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last
    self.indices = list(range(len(dataset)))
    if shuffle:
      np.random.shuffle(self.indices)
    self.batch_idx = 0

  def __len__(self) -> int:
    if self.drop_last:
      return len(self.dataset) // self.batch_size
    else:
      return (len(self.dataset) + self.batch_size - 1) // self.batch_size

  def __iter__(self):
    self.batch_idx = 0
    self.indices = list(range(len(self.dataset)))
    if self.shuffle:
      np.random.shuffle(self.indices)
    return self

  def __next__(self) -> tuple[Tensor, Tensor]:
    if self.batch_idx >= len(self):
      raise StopIteration
    batch_indices = self.indices[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
    x = Tensor([self.dataset[i][0].v for i in batch_indices])
    y = Tensor([self.dataset[i][1].v for i in batch_indices])
    self.batch_idx += 1
    return x, y

  def __repr__(self):
    return f"DataLoader(dataset={self.dataset}, batch_size={self.batch_size}, shuffle={self.shuffle}, drop_last={self.drop_last})"

# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

# for batch in train_loader:
#   break
#%%
# for batch in train_loader:
#   print(batch)
#   break

# x, y = next(train_loader)
# x.v.shape, y.v.shape
#%%