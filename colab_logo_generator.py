# ============================================================
# AI Logo Generator - Google Colab Version
# ============================================================
# Run this in Google Colab with GPU runtime (T4 recommended)
# 
# Instructions:
# 1. Go to https://colab.research.google.com/
# 2. Create a new notebook
# 3. Go to Runtime > Change runtime type > Select T4 GPU
# 4. Copy and paste this entire code into a cell
# 5. Run the cell
# ============================================================

# Install dependencies
print("📦 Installing dependencies...")
!pip install -q diffusers transformers accelerate torch torchvision pillow

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import gc

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "stabilityai/sdxl-turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🖥️ Device: {DEVICE}")
print(f"🎨 Model: {MODEL_ID}")

# ============================================================
# Load Model
# ============================================================
print("\n🚀 Loading SDXL-Turbo...")

pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe = pipe.to(DEVICE)

# Memory optimizations (optional for Colab but good practice)
pipe.enable_attention_slicing()

print("✅ Model loaded successfully!")

# ============================================================
# Logo Generation Function
# ============================================================
def generate_logo(prompt, style="modern", num_images=4, seed=None):
    """
    Generate logos with SDXL-Turbo
    
    Args:
        prompt: Description of the logo (e.g., "coffee shop logo")
        style: One of: modern, minimal, vintage, playful, corporate, tech, organic
        num_images: Number of images to generate (1-4)
        seed: Random seed for reproducibility (optional)
    """
    
    # Style enhancements
    style_keywords = {
        "modern": "sleek, contemporary, clean lines, geometric shapes",
        "minimal": "minimalist, simple, elegant, negative space, ultra clean",
        "vintage": "retro, classic, hand-drawn, nostalgic, heritage",
        "playful": "fun, colorful, friendly, rounded shapes, approachable",
        "corporate": "professional, sophisticated, trustworthy, bold, business",
        "tech": "futuristic, digital, innovative, tech-inspired, cutting edge",
        "organic": "natural, flowing, curved lines, handcrafted, earthy"
    }
    
    # Build prompt
    full_prompt = prompt
    if style.lower() in style_keywords:
        full_prompt += f", {style_keywords[style.lower()]}"
    full_prompt += ", modern minimal clean vector logo, flat icon design, high contrast, white background, professional, high resolution, trending on dribbble"
    
    negative_prompt = "blurry, low quality, pixelated, watermark, text, signature, photo realistic, 3d render, complex background"
    
    print(f"\n{'='*60}")
    print(f"🎨 Generating {num_images} logo(s)...")
    print(f"📝 Prompt: {full_prompt[:80]}...")
    print(f"{'='*60}\n")
    
    # Generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    images = []
    for i in range(num_images):
        print(f"⚡ Generating image {i+1}/{num_images}...")
        
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,  # SDXL-Turbo uses 1-4 steps
            guidance_scale=0.0,     # SDXL-Turbo ignores guidance
            width=1024,
            height=1024,
            generator=generator
        ).images[0]
        
        images.append(image)
        print(f"   ✅ Done!")
    
    print(f"\n🎉 Generated {len(images)} logo(s)!")
    return images

# ============================================================
# Helper function to display images
# ============================================================
def show_logos(images):
    """Display generated logos in a grid"""
    from IPython.display import display
    import matplotlib.pyplot as plt
    
    n = len(images)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 8*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Logo {i+1}")
    
    # Hide empty subplots
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================
# USAGE EXAMPLES
# ============================================================
print("\n" + "="*60)
print("🎯 READY TO GENERATE LOGOS!")
print("="*60)
print("""
Usage Examples:

1. Simple generation:
   images = generate_logo("coffee shop logo")
   show_logos(images)

2. With style:
   images = generate_logo("tech startup logo", style="tech")
   show_logos(images)

3. Generate one image:
   images = generate_logo("bakery logo", style="playful", num_images=1)
   show_logos(images)

4. Save images:
   images = generate_logo("fitness gym logo")
   for i, img in enumerate(images):
       img.save(f"logo_{i}.png")

Available styles: modern, minimal, vintage, playful, corporate, tech, organic
""")

# ============================================================
# Try it now!
# ============================================================
# Uncomment the lines below to generate your first logo:

# images = generate_logo("modern tech startup logo", style="tech", num_images=2)
# show_logos(images)
