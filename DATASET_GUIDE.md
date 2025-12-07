# 📊 Dataset Preparation Guide

Complete guide for preparing your logo dataset for training the classifier.

---

## 📥 Option 1: Using Kaggle Datasets

### Recommended Datasets

1. **LLD Logo Dataset (167K images)**
   - URL: https://www.kaggle.com/datasets/lyly99/logos
   - Size: ~5GB
   - Categories: Multiple logo types
   - Quality: High

2. **Logo-2K+ Dataset**
   - URL: https://www.kaggle.com/datasets/omarsalah22/logo-2k
   - Size: ~2GB
   - Categories: Brand logos
   - Quality: Medium-High

3. **FlickrLogos-32 Dataset**
   - URL: http://www.multimedia-computing.de/flickrlogos/
   - Size: ~500MB
   - Categories: 32 famous brands
   - Quality: High

### Download from Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle API credentials
# 1. Go to https://www.kaggle.com/account
# 2. Create New API Token
# 3. Save kaggle.json to ~/.kaggle/

# Download dataset
kaggle datasets download -d lyly99/logos
unzip logos.zip -d data/raw/
```

---

## 🗂️ Option 2: Creating Custom Dataset

### Directory Structure

```
data/raw/train/
├── Technology/
│   ├── tech_logo_001.jpg
│   ├── tech_logo_002.png
│   └── ... (100+ images)
│
├── Food/
│   ├── food_logo_001.jpg
│   └── ... (100+ images)
│
├── Education/
│   ├── edu_logo_001.jpg
│   └── ... (100+ images)
│
├── Sports/
│   └── ... (100+ images)
│
├── Fashion/
│   └── ... (100+ images)
│
├── Healthcare/
│   └── ... (100+ images)
│
├── Finance/
│   └── ... (100+ images)
│
└── Entertainment/
    └── ... (100+ images)
```

### Collection Methods

#### Method 1: Google Images Scraper

```python
# install: pip install google-images-download
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

# Example: Download tech logos
arguments = {
    "keywords": "technology company logo",
    "limit": 200,
    "print_urls": True,
    "output_directory": "data/raw/train/Technology/",
    "image_directory": "",
    "format": "jpg",
    "size": "medium"
}

response.download(arguments)
```

#### Method 2: Bing Image Downloader

```bash
pip install bing-image-downloader
```

```python
from bing_image_downloader import downloader

downloader.download(
    "restaurant logo",
    limit=200,
    output_dir='data/raw/train/Food',
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)
```

#### Method 3: Manual Collection

**Sources:**
- Dribbble (https://dribbble.com/)
- Behance (https://www.behance.net/)
- LogoLounge (https://www.logolounge.com/)
- 99designs (https://99designs.com/)
- Company websites

---

## 🔍 Data Quality Guidelines

### Image Requirements

✅ **DO:**
- Use clear, high-resolution images (300x300px minimum)
- Include diverse styles within each category
- Ensure logos are centered
- Use white or simple backgrounds
- Include both color and monochrome logos
- Mix modern and classic designs

❌ **DON'T:**
- Use blurry or pixelated images
- Include images with heavy watermarks
- Use screenshots with backgrounds
- Include duplicate or very similar logos
- Use images with text-heavy designs

### Recommended Counts per Category

| Quality Level | Images per Category | Total Dataset |
|---------------|---------------------|---------------|
| Minimum       | 100                 | 800           |
| Good          | 300                 | 2,400         |
| Excellent     | 500-1000            | 4,000-8,000   |
| Professional  | 1000+               | 8,000+        |

---

## 🧹 Data Cleaning Script

Save as `clean_dataset.py`:

```python
"""
Clean and validate logo dataset
"""

from PIL import Image
import os
from pathlib import Path
import shutil

def clean_dataset(data_dir, min_size=224):
    """
    Clean dataset by removing invalid images
    
    Args:
        data_dir: Path to train directory
        min_size: Minimum image dimension
    """
    data_path = Path(data_dir)
    removed = 0
    processed = 0
    
    for category_dir in data_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        print(f"\n📁 Processing: {category_dir.name}")
        
        for img_path in category_dir.glob("*"):
            processed += 1
            
            try:
                # Open image
                img = Image.open(img_path)
                
                # Check size
                if min(img.size) < min_size:
                    print(f"   ❌ Too small: {img_path.name} ({img.size})")
                    img_path.unlink()
                    removed += 1
                    continue
                
                # Check format
                if img.format not in ['JPEG', 'PNG', 'JPG']:
                    print(f"   ❌ Invalid format: {img_path.name}")
                    img_path.unlink()
                    removed += 1
                    continue
                
                # Verify image can be loaded
                img.verify()
                
            except Exception as e:
                print(f"   ❌ Error: {img_path.name} - {e}")
                img_path.unlink()
                removed += 1
    
    print(f"\n✅ Processed: {processed} images")
    print(f"❌ Removed: {removed} images")
    print(f"✅ Remaining: {processed - removed} images")

if __name__ == "__main__":
    clean_dataset("data/raw/train")
```

Run:
```bash
python clean_dataset.py
```

---

## 📊 Data Augmentation

The training script automatically applies:

- ✅ Random rotation (±20°)
- ✅ Width/height shifts (20%)
- ✅ Horizontal flips
- ✅ Zoom (20%)
- ✅ Brightness adjustments

No manual augmentation needed!

---

## 🔄 Data Splitting

The training script uses automatic 80/20 split:
- 80% training
- 20% validation

For custom split, organize as:

```
data/raw/
├── train/
│   ├── Technology/
│   └── ...
└── test/
    ├── Technology/
    └── ...
```

---

## 📈 Dataset Statistics Script

Save as `dataset_stats.py`:

```python
"""
Generate dataset statistics
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def analyze_dataset(data_dir):
    """Analyze and visualize dataset"""
    data_path = Path(data_dir)
    
    categories = {}
    sizes = []
    
    for category_dir in data_path.iterdir():
        if not category_dir.is_dir():
            continue
        
        count = len(list(category_dir.glob("*")))
        categories[category_dir.name] = count
        
        # Sample sizes
        for img_path in list(category_dir.glob("*"))[:10]:
            try:
                img = Image.open(img_path)
                sizes.append(min(img.size))
            except:
                pass
    
    # Plot distribution
    plt.figure(figsize=(12, 5))
    
    # Category counts
    plt.subplot(1, 2, 1)
    plt.bar(categories.keys(), categories.values())
    plt.title('Images per Category')
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    
    # Size distribution
    plt.subplot(1, 2, 2)
    plt.hist(sizes, bins=20)
    plt.title('Image Size Distribution')
    plt.xlabel('Min Dimension (px)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('dataset_stats.png', dpi=300)
    print(f"✅ Stats saved to dataset_stats.png")
    
    # Print summary
    print("\n📊 Dataset Summary:")
    print(f"   Total categories: {len(categories)}")
    print(f"   Total images: {sum(categories.values())}")
    print(f"   Avg per category: {sum(categories.values()) / len(categories):.0f}")
    print(f"   Min size: {min(sizes)}px")
    print(f"   Max size: {max(sizes)}px")
    print(f"   Avg size: {sum(sizes)/len(sizes):.0f}px")

if __name__ == "__main__":
    analyze_dataset("data/raw/train")
```

---

## ✅ Dataset Checklist

Before training, verify:

- [ ] All categories have 100+ images
- [ ] Images are clear and properly centered
- [ ] No corrupt or invalid files
- [ ] Diverse styles within each category
- [ ] Proper directory structure
- [ ] Images are at least 224x224px
- [ ] Mix of color and monochrome logos
- [ ] No duplicate images

---

## 🎯 Category Guidelines

### Technology
- Software companies
- Tech startups
- IT services
- Electronics brands
- Digital platforms

### Food & Restaurant
- Restaurants
- Food brands
- Cafes
- Food delivery
- Beverages

### Education
- Schools
- Universities
- Online learning
- Educational apps
- Training centers

### Sports & Fitness
- Gyms
- Sports teams
- Fitness apps
- Athletic brands
- Sports equipment

### Fashion & Beauty
- Clothing brands
- Cosmetics
- Accessories
- Jewelry
- Fashion retailers

### Healthcare
- Hospitals
- Medical devices
- Pharmaceutical
- Health apps
- Wellness centers

### Finance
- Banks
- Fintech
- Insurance
- Investment firms
- Crypto platforms

### Entertainment
- Media companies
- Gaming
- Music platforms
- Streaming services
- Event organizers

---

## 🚀 Quick Start

```bash
# 1. Create directories
mkdir -p data/raw/train/{Technology,Food,Education,Sports,Fashion,Healthcare,Finance,Entertainment}

# 2. Add images to each category folder

# 3. Clean dataset
python clean_dataset.py

# 4. Check statistics
python dataset_stats.py

# 5. Train model
python src/train_classifier.py
```

---

**Need help? Check the main README.md or open an issue!**