import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------
# 1. Device (CPU/GPU)
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------
# 2. Dummy dataset
# ----------------------
# Random images: (batch_size, channels, height, width)
X = torch.randn(64, 3, 32, 32).to(device)      # 64 fake images
y = torch.randint(0, 10, (64,)).to(device)     # 10 fake classes

# ----------------------
# 3. Simple CNN model
# ----------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16×16×16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32×8×8
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleCNN().to(device)

# ----------------------
# 4. Loss + optimizer
# ----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------------
# 5. Training loop
# ----------------------
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")
