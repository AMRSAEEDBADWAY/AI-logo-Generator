import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm

# =========================
# CPU Optimization
# =========================
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False

# =========================
# PATHS
# =========================
TRAIN_DIR = r"D:\ai-logo-generator\data\datasetcopy\trainandtest\train"
TEST_DIR = r"D:\ai-logo-generator\data\datasetcopy\trainandtest\test"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cpu")
print("Using:", device)

# =========================
# LIGHT TRANSFORMS
# =========================
IMG_SIZE = 128   # مناسب للأجهزة الضعيفة

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)

# =========================
# MOBILE NET V2 (خفيف جدًا)
# =========================
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# =========================
# TRAINING
# =========================
def train(epochs=5):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()

        correct = 0
        total = 0
        running_loss = 0

        for imgs, labels in tqdm(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Train Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "light_model.pth"))
    print("✔ Model saved!")

train(epochs=5)

# =========================
# TEST + CSV RESULTS
# =========================
model.eval()

all_preds = []
all_probs = []
true_labels = []
file_paths = [path for path, _ in test_dataset.imgs]

with torch.no_grad():
    for imgs, labels in tqdm(test_loader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        true_labels.extend(labels.numpy())

rows = []
for i in range(len(true_labels)):
    rows.append({
        "image": file_paths[i],
        "true_label": class_names[true_labels[i]],
        "predicted_label": class_names[all_preds[i]],
        "confidence": float(np.max(all_probs[i]))
    })

df = pd.DataFrame(rows)
df.to_csv("models/test_predictions.csv", index=False)
print("✔ CSV saved → models/test_predictions.csv")
