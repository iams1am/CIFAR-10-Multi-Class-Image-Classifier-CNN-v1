import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. FORCE CUDA & VRAM SAFETY
# This automatically detects your MX330 and forces the script to use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache() # Clears out any lingering memory

# 2. DATA PREPARATION
# We normalize the images to a range of [-1, 1] for better training stability
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# BATCH SIZE: This is your #1 defense against Out Of Memory (OOM) errors.
# 64 images at a time will comfortably fit inside your 2GB limit.
BATCH_SIZE = 64 

print("\nDownloading and loading CIFAR-10 dataset (approx 160MB)...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 3. DEFINE THE NEURAL NETWORK (A Simple CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # The Feature Extractor (Looks for edges, shapes, and textures)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Shrinks 32x32 image to 16x16
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Shrinks to 8x8
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Shrinks to 4x4
        )
        # The Classifier (Takes the found features and guesses the class)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10) # 10 final output classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 4 * 4) # Flattens the 3D tensor into a 1D vector
        x = self.classifier(x)
        return x

# Push the model architecture directly into the MX330's VRAM
model = SimpleCNN().to(device)

# 4. LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. THE TRAINING LOOP
EPOCHS = 5
print("\nStarting Training Loop...")

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Move the batch of images and their labels to GPU VRAM
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the gradients so old math doesn't pile up in VRAM
        optimizer.zero_grad()

        # Forward pass (predict) -> Calculate Error (loss) -> Backward pass (learn) -> Step (update weights)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Print an update every 200 batches to watch it learn
        if i % 200 == 199:
            vram_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"[Epoch {epoch + 1}, Batch {i + 1:4d}] Loss: {running_loss / 200:.3f} | VRAM Peak: {vram_used:.1f} MB")
            running_loss = 0.0

print("\nFinished Training!")

# 6. TEST THE NETWORK
print("\nTesting model accuracy on 10,000 unseen images...")
correct = 0
total = 0

# Tell PyTorch we aren't training. This skips gradient calculations and saves massive VRAM.
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nFinal Accuracy on 10,000 test images: {100 * correct / total:.2f}%")
# Save the learned weights to a file named 'cifar_net.pth'
torch.save(model.state_dict(), 'cifar_net.pth')
print("Model saved successfully to cifar_net.pth!")