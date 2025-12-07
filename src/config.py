"""
Configuration file for AI Logo Generator
Contains all settings, paths, and hyperparameters
"""

import os
from pathlib import Path

# ========== PROJECT PATHS ==========
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LORA_DIR = MODELS_DIR / "lora"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LORA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ========== CLASSIFIER SETTINGS ==========
CLASSIFIER_CONFIG = {
    "model_path": MODELS_DIR / "best_model.pth",
    "image_size": (224, 224),
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    
    # Logo Categories
    "classes": [
        "Technology",
        "Food & Restaurant",
        "Education",
        "Sports & Fitness",
        "Fashion & Beauty",
        "Healthcare",
        "Finance",
        "Entertainment"
    ]
}

# ========== LOGO GENERATION SETTINGS ==========
GENERATION_CONFIG = {
    "model_id": "stabilityai/sdxl-turbo",
    "local_model_path": MODELS_DIR / "sdxl-turbo-local",
    "image_size": 512,  # SDXL-Turbo native resolution
    "base_prompt_suffix": "vector logo, flat design, clean sharp edges, high quality",
    "negative_prompt": "blurry, fuzzy, melted, distorted, pixelated, low resolution, photo, realistic",
    # 4 steps for best quality/speed balance on Turbo
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "num_images": 4,
    "use_lora": False,
    "lora_scale": 0.8
}

# ========== DATA PROCESSING ==========
DATA_CONFIG = {
    "train_dir": Path(r"D:\ai-logo-generator\data\datasetcopy\trainandtest\train"),
    "test_dir": DATA_DIR / "raw" / "test",
    "processed_dir": DATA_DIR / "processed",
    "augmentation": {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "horizontal_flip": True,
        "zoom_range": 0.2,
        "fill_mode": "nearest"
    }
}

# ========== API KEYS (use environment variables) ==========
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# ========== STREAMLIT UI SETTINGS ==========
UI_CONFIG = {
    "page_title": "AI Logo Generator & Classifier",
    "page_icon": "🎨",
    "layout": "wide",
    "theme": "dark"
}