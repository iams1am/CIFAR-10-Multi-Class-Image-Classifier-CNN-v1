import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")
if device.type == 'cuda':
    torch.cuda.empty_cache()

# 1. DATA AUGMENTATION (Train set only)
# We add RandomCrop and RandomHorizontalFlip to artificially expand our dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test data should NEVER be augmented. We want to test on pure, unaltered images.
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

BATCH_SIZE = 64 # Still totally safe for 2GB VRAM

print("\nLoading datasets...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# 2. ADVANCED ARCHITECTURE (Deeper, BatchNorm, Dropout)
class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2), # Drop 20% of neurons

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3), # Drop 30% of neurons

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)  # Drop 40% of neurons
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5), # Drop 50% of dense neurons
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
        return x

model = AdvancedCNN().to(device)

criterion = nn.CrossEntropyLoss()
# Added weight_decay (L2 regularization) for even more overfitting prevention
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning Rate Scheduler: Drops the learning rate by 90% after 15 epochs to "fine-tune" the details
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# 3. TRAINING LOOP
EPOCHS = 20
print(f"\nStarting Advanced Training Loop for {EPOCHS} Epochs...")

for epoch in range(EPOCHS):
    model.train() # CRITICAL: Tells BatchNorm and Dropout that we are training
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    # Print summary at the end of each epoch
    vram_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Epoch {epoch + 1:2d}/{EPOCHS} | Avg Loss: {running_loss / len(trainloader):.3f} | VRAM Peak: {vram_used:.1f} MB | LR: {scheduler.get_last_lr()[0]:.5f}")
    
    scheduler.step() # Update learning rate

print("\nFinished Training!")
torch.save(model.state_dict(), 'cifar_advanced.pth')
print("Model saved to cifar_advanced.pth")

# 4. TESTING
print("\nTesting advanced model accuracy...")
model.eval() # CRITICAL: Turns OFF Dropout and freezes BatchNorm for testing
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Accuracy on 10,000 test images: {100 * correct / total:.2f}%")