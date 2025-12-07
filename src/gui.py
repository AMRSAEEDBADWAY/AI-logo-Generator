import os
import json
import torch
import torch.nn as nn
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ================================
# CONFIG
# ================================
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CLASS_JSON = os.path.join(MODEL_DIR, "class_indices.json")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# LOAD CLASSES
# ================================
with open(CLASS_JSON, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(idx_to_class)

# ================================
# MODEL
# ================================
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ================================
# IMAGE TRANSFORM
# ================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================================
# GUI
# ================================
root = tk.Tk()
root.title("AI Logo Classifier")
root.geometry("600x600")
root.resizable(False, False)

label_image = tk.Label(root)
label_image.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

# ================================
# PREDICT FUNCTION
# ================================
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((300, 300))
    
    tk_img = ImageTk.PhotoImage(img_resized)
    label_image.config(image=tk_img)
    label_image.image = tk_img

    tensor_img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor_img)
        probs = torch.softmax(output, dim=1)
        conf, pred_class = torch.max(probs, dim=1)

    class_name = idx_to_class[pred_class.item()]
    confidence = float(conf.item()) * 100

    result_label.config(text=f"Prediction: {class_name}\nConfidence: {confidence:.2f}%")

# ================================
# BUTTON: OPEN IMAGE
# ================================
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if file_path:
        predict_image(file_path)

btn = tk.Button(root, text="اختر صورة للتجربة", font=("Arial", 18), command=open_file)
btn.pack(pady=20)

root.mainloop()
