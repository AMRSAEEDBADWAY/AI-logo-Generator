import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================================
# CONFIG
# ================================
TEST_DIR = r"D:\ai-logo-generator\data\datasetcopy\trainandtest\test"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32

print("Using:", DEVICE)

# ================================
# LOAD CLASS NAMES
# ================================
with open(os.path.join(MODEL_DIR, "class_indices.json"), "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(idx_to_class)
print("Classes:", idx_to_class)

# ================================
# TEST TRANSFORMS
# ================================
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================================
# LOAD TEST DATA
# ================================
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Found {len(test_dataset)} test images.")

# ================================
# LOAD MODEL
# ================================
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

print("Loading EfficientNet-B0...")
model = efficientnet_b0(weights=None)  # No IMAGENET weights, we load ours
model.classifier[1] = nn.Linear(1280, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ================================
# TESTING
# ================================
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ================================
# RESULTS
# ================================
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\n🔥 Test Accuracy: {accuracy:.4f}")

print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=list(idx_to_class.values())))

# ================================
# CONFUSION MATRIX
# ================================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(idx_to_class.values()),
            yticklabels=list(idx_to_class.values()))
plt.title("Test Confusion Matrix")
plt.savefig(os.path.join(MODEL_DIR, "test_confusion_matrix.png"))
plt.close()

print("\n✔ All test results saved inside:", MODEL_DIR)
