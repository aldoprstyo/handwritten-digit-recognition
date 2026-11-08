# Skrip opsional untuk melatih ulang model dan menghasilkan digit_model.pth
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.network(x)

digits = load_digits()
X = digits.data.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = DigitModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()
loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=64, shuffle=True
)
for epoch in range(100):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
model.eval()
with torch.no_grad():
    acc = (model(torch.tensor(X_test)).argmax(1).numpy() == y_test).mean()
print(f"Akurasi data uji: {acc:.4f}")
torch.save(model.state_dict(), "digit_model.pth")
print("Model tersimpan: digit_model.pth")
