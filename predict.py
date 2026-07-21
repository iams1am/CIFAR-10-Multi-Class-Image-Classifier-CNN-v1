import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. DEFINE THE EXACT SAME MODEL ARCHITECTURE
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.classifier(x)
        return x

# 2. LOAD THE SAVED WEIGHTS
model = SimpleCNN().to(device)
try:
    model.load_state_dict(torch.load('cifar_net.pth', weights_only=True))
    print("Successfully loaded 'cifar_net.pth' onto GPU!")
except FileNotFoundError:
    print("Error: Could not find 'cifar_net.pth'. Did you save the model weights?")
    exit()

model.eval() # Set model to evaluation mode

# 3. PREPARE DATASET
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 4. PICK 5 RANDOM TEST IMAGES AND PREDICT
print("\n--- TEST PREDICTIONS ---")
for i in range(5):
    # Pick a random image from the 10,000 test set
    idx = random.randint(0, len(testset) - 1)
    img, actual_label = testset[idx]
    
    # Add a batch dimension (shape becomes [1, 3, 32, 32]) and push to GPU
    img_tensor = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        # Get probabilities using Softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    actual_name = classes[actual_label]
    predicted_name = classes[predicted_class.item()]
    conf_percent = confidence.item() * 100
    
    match_status = "CORRECT" if actual_name == predicted_name else "WRONG"
    print(f"Sample {i+1}: Actual = {actual_name:<8} | Predicted = {predicted_name:<8} ({conf_percent:.1f}% confidence) [{match_status}]")