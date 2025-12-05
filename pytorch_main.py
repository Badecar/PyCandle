
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

# --- HYPERPARAMETERS ---
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 1

# --- DATA LOADING ---
# Matches the normalization / 255.0
transform = transforms.Compose([
    transforms.ToTensor(), # Converts to [0, 1] (which is / 255.0)
])

# Download and load training data
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Download and load test data
test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# --- MODEL DEFINITION ---
class PyTorchModel(nn.Module):
    def __init__(self, num_classes=10):
        super(PyTorchModel, self).__init__()
        self.layers = nn.Sequential(
            # Conv2D(in_channels=1, num_kernels=32, kernel_size=3, stride=1, padding="same")
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            
            # Conv2D(in_channels=32, num_kernels=32, kernel_size=3, stride=1, padding="same")
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            
            # MaxPool2D(2,2)
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14
            
            # Conv2D(in_channels=32, num_kernels=64, kernel_size=3, stride=1, padding="same")
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            
            # Conv2D(in_channels=64, num_kernels=64, kernel_size=3, stride=1, padding="same")
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            
            # MaxPool2D(2,2)
            nn.MaxPool2d(kernel_size=2, stride=2), # 7x7
            
            nn.Flatten(),
            
            # Linear(n_in=64*7*7, n_out=512, bias=True)
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            
            # Linear(n_in=512, n_out=num_classes, bias=True)
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = PyTorchModel(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- TRAINING LOOP ---
loss_list = []

print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        loss_val = loss.item()
        loss_list.append(loss_val)
        
        # Calculate training accuracy for the batch
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        acc = correct / total
        
        print(f"Epoch {epoch}, Batch {i}, Loss: {loss_val:.4f}, Acc: {acc:.4f}")

# --- EVALUATION ---
print("Starting evaluation...")
model.eval()
y_true = []
y_pred = []
total_loss = 0
batches = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        batches += 1
        
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Metrics
avg_loss = total_loss / batches
accuracy = (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Validation Loss: {avg_loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation F1 Score (macro): {f1:.4f}")

# --- PLOTTING ---
plt.figure(figsize=(10, 5))
plt.plot(loss_list)
plt.title("Training Loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.show()

