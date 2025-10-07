# chest_xray_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===============================
# CONFIG
# ===============================
data_dir = "./chest_xray/chest_xray"   # root folder with train/ val/ test/
batch_size = 32
epochs = 30
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "cnn_chest_xray_light.pth"

# ===============================
# TRANSFORMS
# ===============================
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# DATASET LOAD (use provided split)
# ===============================
train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform_train)
val_dataset   = datasets.ImageFolder(f"{data_dir}/val",   transform=transform_test)
test_dataset  = datasets.ImageFolder(f"{data_dir}/test",  transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes  # ["NORMAL", "PNEUMONIA"]

# ===============================
# MODEL (Small CNN)
# ===============================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> [16,64,64]

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> [32,32,32]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> [64,16,16]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SmallCNN(num_classes=len(class_names)).to(device)

# ===============================
# LOSS & OPTIMIZER
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ===============================
# EARLY STOPPING
# ===============================
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStopping(patience=5)

# ===============================
# TRAIN / EVAL LOOP
# ===============================
def evaluate(loader, net):
    net.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, 100. * correct / total

print("Starting training...")
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = 100. * correct / total

    val_loss, val_acc = evaluate(val_loader, model)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss {train_loss:.4f} Acc {train_acc:.2f}% "
          f"| Val Loss {val_loss:.4f} Acc {val_acc:.2f}%")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_path)
        print("Saved new best model")

    # Early stopping
    early_stopper(val_loss)
    if early_stopper.early_stop:
        print("⏹️ Early stopping triggered")
        break

# ===============================
# FINAL TEST
# ===============================
print("Loading best model...")
model.load_state_dict(torch.load(model_path))
test_loss, test_acc = evaluate(test_loader, model)
print(f"Test Loss {test_loss:.4f} | Test Acc {test_acc:.2f}%")
