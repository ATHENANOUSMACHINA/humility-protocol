import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("HUMILITY PROTOCOL - MNIST TEST")
print("="*50)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

transform = transforms.ToTensor()
train_data = datasets.MNIST('.', train=True, download=True, 
transform=transform)
test_data = datasets.MNIST('.', train=False, download=True, 
transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

print("\nTraining baseline model...")
model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(2):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: 
{loss.item():.4f}")

print("\nTesting...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

accuracy = 100. * correct / total
print(f"\nBaseline Accuracy: {accuracy:.2f}%")
print("Done!")
