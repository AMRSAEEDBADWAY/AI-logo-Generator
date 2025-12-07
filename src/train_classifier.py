import os
import json
from config import DATA_CONFIG
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import random

# ======================================
# CONFIG
# ======================================
TRAIN_DIR = DATA_CONFIG['train_dir']
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 15  # ارفع لـ 20 لو عايز accuracy = 90%
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using:", DEVICE)
torch.backends.cudnn.benchmark = True
if DEVICE.type == "cuda":
    print("GPU Name:", torch.cuda.get_device_name(0))
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    torch.cuda.empty_cache()

USE_AMP = True
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

# ======================================
# EARLY STOPPING CLASS
# ======================================
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def check(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False  # لا توقف

        # مفيش تحسن كفاية
        if current_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"⚠️ No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                print("⛔ Early Stopping Triggered — Training Stopped.")
                return True  # وقف التدريب
        else:
            self.best_score = current_score
            self.counter = 0

        return False


# ======================================
# DATA AUGMENTATION
# ======================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.3),
    transforms.RandomAffine(10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================================
# MIXUP & CUTMIX
# ======================================
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size).to(x.device)

    cx = np.random.randint(w)
    cy = np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, w)
    y2 = min(cy + cut_h // 2, h)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1)) / (w * h)

    return x, y, y[index], lam


# ======================================
# MAIN EXECUTION
# ======================================
if __name__ == '__main__':

    dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    num_classes = len(dataset.classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Save classes
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(dataset.class_to_idx, f)

    # ======================================
    # MODEL (EfficientNet-B0)
    # ======================================
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    print("Loading EfficientNet-B0...")

    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    for p in backbone.parameters():
        p.requires_grad = True  # Full fine-tuning

    backbone.classifier[1] = nn.Linear(1280, num_classes)
    model = backbone.to(DEVICE)

    # ======================================
    # CLASS WEIGHTS
    # ======================================
    targets = np.array(dataset.targets)
    train_targets = targets[train_dataset.indices]

    class_counts = np.bincount(train_targets)
    class_weights = len(train_targets) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ======================================
    # OPTIMIZER + SCHEDULER
    # ======================================
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    early_stop = EarlyStopping(patience=3, min_delta=0.001)

    best_val_acc = 0

    # ======================================
    # TRAIN LOOP
    # ======================================
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()

        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            # MixUp / CutMix
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)
            else:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=1.0)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(imgs)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ====================
        # VALIDATION
        # ====================
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = torch.argmax(model(imgs), 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Validation Accuracy = {val_acc:.4f}")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
            print("✔ Saved Best Model!")

        # ⭐ EarlyStopping
        if early_stop.check(val_acc):
            break

    print("\n🎉 Training complete.")
