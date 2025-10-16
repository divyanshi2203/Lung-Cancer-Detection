# ================================
# Lung Cancer Detection Training Script
# ================================

# --- 1. IMPORTS ---
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ================================
# --- 2. SETTINGS ---
# ================================

# Make sure the classifier folder exists for saving the model
os.makedirs("classifier", exist_ok=True)

# Device configuration - GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image and training settings
IMAGE_SIZE = 64      # Resize images to 64x64
BATCH_SIZE = 2       # Small batch size since you only have 2 images
EPOCHS = 10          # You can increase when you have more data
LEARNING_RATE = 0.001

# ================================
# --- 3. DATA LOADING ---
# ================================

# Transforms: resize, convert to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load training dataset
train_dir = 'data/train'  # <-- Make sure your images are here
train_data = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Show class mapping
print("Class to index mapping:", train_data.class_to_idx)
# Example output: {'cancer': 0, 'normal': 1}

# ================================
# --- 4. DEFINE THE CNN MODEL ---
# ================================
class CancerNet(nn.Module):
    def __init__(self):
        super(CancerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 13 * 13, 128)  # 13*13 is output size after conv+pool
        self.fc2 = nn.Linear(128, 2)  # 2 classes: cancer, normal

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model
model = CancerNet().to(device)
print(model)

# ================================
# --- 5. LOSS FUNCTION & OPTIMIZER ---
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================================
# --- 6. TRAINING LOOP ---
# ================================
print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

print("Training complete!")

# ================================
# --- 7. SAVE THE MODEL ---
# ================================
torch.save(model.state_dict(), "classifier/cancer_detector.pth")
print("Model saved successfully at: classifier/cancer_detector.pth")
