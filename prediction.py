import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam

# 20 samples
n = 20

# Generate some synthetic features
feature1 = np.linspace(0, 10, n)
feature2 = np.random.uniform(0, 5, n)

# Define a linear relationship with noise
noise = np.random.normal(0, 2, n)  # Gaussian noise with std=2
y = 3 * feature1 + 2 * feature2 + noise

# DataFrame
df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'y': y
})

X = df[['feature1', 'feature2']]
y = df['y']
ratio = int(0.8*len(df))
X_train, X_test, y_train, y_test = X[:ratio], X[ratio:], y[:ratio], y[ratio:]

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Create dataset
dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
dataset_test = TensorDataset(X_test_tensor, y_test_tensor)

# Wrap in DataLoader
batch_size = 32
loader_train = DataLoader(dataset_train, batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size, shuffle=False)

class LinearReplicator(nn.Module):
    def __init__(self, in_features = 2):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)


lr = 0.0001
num_epochs = 20

model = LinearReplicator(in_features=2)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=lr)


for epoch in range(1, num_epochs+1):
    model.train()
    epoch_loss = 0.0

    for xb, yb in loader_train:
          y_pred = model(xb)
          loss = criterion(y_pred, yb)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

    epoch_loss /= len(loader_train.dataset)

    if epoch == 1 or epoch % 50 == 0:
         print(epoch_loss)


model.eval()
with torch.no_grad():
     all_preds = []
     all_targets = []
     for xb, yb in loader_test:
          test_pred = model(xb)
          test_loss = criterion(test_pred, yb)
          all_preds.append(test_pred)
          all_targets.append(yb)

     y_hat = torch.vstack(all_preds)
     y_true = torch.vstack(all_targets)

mse_test = criterion(y_hat, y_true).item()
print(mse_test)

import matplotlib.pyplot as plt

plt.plot(y_hat)
plt.plot(y_true)
plt.show()

