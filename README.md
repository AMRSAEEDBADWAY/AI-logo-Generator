# 🎨 AI Logo Generator + Classifier

A complete AI-powered logo generation and classification system built with deep learning. Generate professional logos from text descriptions and classify existing logos into categories.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red.svg)

---

## 🌟 Features

### Logo Generation
- ✨ **AI-Powered Creation**: Generate logos using SDXL or FLUX models
- 🎨 **Multiple Styles**: Modern, minimal, vintage, playful, corporate, tech, organic
- 🔧 **Advanced Customization**: Custom prompts, negative prompts, style mixing
- 📥 **Easy Export**: Download as PNG with one click
- 🎭 **Background Removal**: Automatic background removal for transparent logos
- 🌈 **Color Analysis**: Extract dominant color palettes

### Logo Classification
- 🔍 **Automatic Categorization**: AI predicts logo category
- 📊 **Confidence Scores**: See prediction confidence for all categories
- 🎯 **8 Categories**: Technology, Food, Education, Sports, Fashion, Healthcare, Finance, Entertainment
- 🧠 **Custom CNN**: Trained deep learning model

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster generation)
- 8GB+ RAM
- HuggingFace account (for SDXL/FLUX access)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/ai-logo-generator.git
cd ai-logo-generator
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up HuggingFace Token

1. Create account at [HuggingFace](https://huggingface.co/)
2. Generate access token from [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Accept model licenses:
   - [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
   - [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-schnell) (optional)

4. Set environment variable:

```bash
# Windows (Command Prompt)
set HUGGINGFACE_TOKEN=your_token_here

# Windows (PowerShell)
$env:HUGGINGFACE_TOKEN="your_token_here"

# Linux/Mac
export HUGGINGFACE_TOKEN="your_token_here"
```

Or create a `.env` file:

```bash
HUGGINGFACE_TOKEN=your_token_here
```

---

## 🎓 Training the Classifier

### Step 1: Prepare Dataset

Organize your dataset as follows:

```
data/raw/train/
├── Technology/
│   ├── logo1.jpg
│   ├── logo2.png
│   └── ...
├── Food/
│   ├── logo1.jpg
│   └── ...
├── Education/
│   └── ...
└── Sports/
    └── ...
```

**Recommended Datasets:**
- [Kaggle Logo Dataset](https://www.kaggle.com/datasets/lyly99/logos)
- [LLD Logo Dataset](https://data.vision.ee.ethz.ch/cvl/lld/)
- Or create your own custom dataset

**Dataset Requirements:**
- Minimum 100 images per category
- Image formats: JPG, PNG
- Recommended: 500-1000 images per category for best results

### Step 2: Configure Training

Edit `src/config.py` if needed:

```python
CLASSIFIER_CONFIG = {
    "image_size": (224, 224),      # Input size
    "batch_size": 32,               # Batch size
    "epochs": 50,                   # Training epochs
    "learning_rate": 0.001,         # Learning rate
    "validation_split": 0.2,        # Validation split
}
```

### Step 3: Train Model

```bash
python src/train_classifier.py
```

**Training Output:**
- `models/logo_classifier.h5` - Trained model
- `models/class_indices.json` - Class mappings
- `models/training_history.png` - Training plots
- `models/confusion_matrix.png` - Confusion matrix

**Training Time:**
- CPU: 2-4 hours for 50 epochs
- GPU: 30-60 minutes for 50 epochs

**For Google Colab Training:**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Upload dataset to Drive
# Run training
!python src/train_classifier.py
```

---

## 🚀 Running the Application

### Start Streamlit App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using Docker (Optional)

```bash
# Build image
docker build -t ai-logo-generator .

# Run container
docker run -p 8501:8501 -e HUGGINGFACE_TOKEN=your_token ai-logo-generator
```

---

## 📱 Usage Guide

### Generating Logos

1. **Navigate** to "Generate Logo" page
2. **Describe** your logo in the text box:
   ```
   Example: "Modern coffee shop logo with minimalist coffee cup, 
   warm brown and cream colors, elegant and simple"
   ```
3. **Choose Style**: Select from presets (modern, minimal, vintage, etc.)
4. **Advanced Options** (optional):
   - Custom prompt suffix
   - Negative prompt (things to avoid)
   - Number of images (1-6)
   - Random seed for reproducibility

5. **Generate**: Click "Generate Logo" button
6. **Download**: Save your favorites as PNG

### Tips for Better Logos:
- Be specific about your industry/business
- Mention preferred colors
- Describe the mood/feeling
- Include style keywords (minimal, bold, elegant, etc.)
- Use negative prompts to avoid unwanted elements

### Classifying Logos

1. **Navigate** to "Classify Logo" page
2. **Upload** logo image (PNG/JPG)
3. **Classify**: Click "Classify Logo" button
4. **Review** predictions and confidence scores

---

## 🎨 Advanced Features

### Using LoRA (Fine-tuned Models)

1. Train or download a logo-specific LoRA
2. Place LoRA file in `models/lora/`
3. Enable in `src/config.py`:

```python
GENERATION_CONFIG = {
    "use_lora": True,
    "lora_path": LORA_DIR / "logo_lora.safetensors",
    "lora_scale": 0.8
}
```

### Background Removal

After generating logos:
1. Click "Remove BG" button under any logo
2. Preview result
3. Download PNG with transparent background

### Color Palette Extraction

1. Click "Colors" button under any generated logo
2. View 5 dominant colors
3. Use for brand consistency

---

## 📂 Project Structure

```
ai-logo-generator/
│
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env                        # Environment variables (create this)
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── utils.py               # Helper functions
│   ├── train_classifier.py   # CNN training script
│   └── generate_logo.py      # Logo generation module
│
├── models/
│   ├── logo_classifier.h5    # Trained CNN (generated)
│   ├── class_indices.json    # Class mappings (generated)
│   └── lora/                 # LoRA weights folder
│
├── data/
│   ├── raw/                  # Raw dataset
│   │   ├── train/
│   │   └── test/
│   └── processed/            # Preprocessed data
│
└── notebooks/
    └── training_analysis.ipynb  # Training visualization
```

---

## ⚙️ Configuration

All settings are in `src/config.py`:

### Classifier Settings
```python
CLASSIFIER_CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
}
```

### Generator Settings
```python
GENERATION_CONFIG = {
    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "num_images": 4,
}
```

---

## 🐛 Troubleshooting

### Issue: "Out of Memory" Error

**Solution:**
```python
# In src/config.py, reduce:
CLASSIFIER_CONFIG['batch_size'] = 16  # Instead of 32
GENERATION_CONFIG['num_images'] = 2   # Instead of 4
```

### Issue: "Model not found" Error

**Solution:**
```bash
# Train classifier first
python src/train_classifier.py

# Or download pre-trained model
# (if available from project releases)
```

### Issue: Slow Generation

**Solutions:**
- Use GPU instead of CPU
- Reduce `num_inference_steps` to 20
- Use FLUX instead of SDXL (faster)
- Generate fewer images at once

### Issue: "Invalid HuggingFace Token"

**Solution:**
1. Verify token is correct
2. Accept model license on HuggingFace
3. Check token has read permissions

### Issue: Background Removal Not Working

**Solution:**
```bash
# Reinstall rembg
pip uninstall rembg
pip install rembg
```

---

## 🔧 System Requirements

### Minimum
- **CPU:** Intel i5 or equivalent
- **RAM:** 8GB
- **Storage:** 10GB free space
- **OS:** Windows 10, Linux, macOS

### Recommended
- **CPU:** Intel i7/AMD Ryzen 7 or better
- **RAM:** 16GB+
- **GPU:** NVIDIA GPU with 6GB+ VRAM
- **Storage:** 20GB+ free space (for models)

---

## 📊 Model Performance

### Classifier Metrics (Example)
- **Accuracy:** 85-92% (depends on dataset)
- **F1 Score:** 0.83-0.90
- **Training Time:** 30-60 minutes (GPU)

### Generator Performance
- **Generation Time:** 
  - SDXL: 10-30 seconds per image (GPU)
  - FLUX: 5-15 seconds per image (GPU)
  - CPU: 2-5 minutes per image

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

**Note:** Model licenses:
- SDXL: [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- FLUX: Check [FLUX license](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

---

## 🙏 Acknowledgments

- **Stability AI** - Stable Diffusion XL
- **Black Forest Labs** - FLUX
- **HuggingFace** - Diffusers library
- **TensorFlow** - Deep learning framework
- **Streamlit** - Web framework

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/ai-logo-generator/issues)
- **Documentation:** See `/docs` folder
- **Email:** your.email@example.com

---

## 🎯 Future Enhancements

- [ ] Vector SVG export
- [ ] Batch generation
- [ ] Logo editing tools
- [ ] Style transfer
- [ ] Mobile app
- [ ] API endpoints
- [ ] Logo animation

---

**Happy Logo Creating! 🎨✨**