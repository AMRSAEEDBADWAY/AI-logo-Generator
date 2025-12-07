"""
Utility functions for image processing, color extraction, and helpers
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import List, Tuple
import cv2

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    from colorthief import ColorThief
    COLORTHIEF_AVAILABLE = True
except ImportError:
    COLORTHIEF_AVAILABLE = False


def preprocess_image_for_classifier(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Preprocess image for CNN classifier
    
    Args:
        image: PIL Image
        target_size: Target dimensions (width, height)
    
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def remove_background(image: Image.Image) -> Image.Image:
    """
    Remove background from logo image
    
    Args:
        image: Input PIL Image
    
    Returns:
        Image with transparent background
    """
    if not REMBG_AVAILABLE:
        print("Warning: rembg not installed. Returning original image.")
        return image
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Remove background
    output = remove(img_byte_arr)
    
    # Convert back to PIL Image
    return Image.open(io.BytesIO(output))


def extract_dominant_colors(image: Image.Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
    """
    Extract dominant colors from an image
    
    Args:
        image: PIL Image
        num_colors: Number of dominant colors to extract
    
    Returns:
        List of RGB tuples
    """
    if not COLORTHIEF_AVAILABLE:
        print("Warning: colorthief not installed.")
        return [(255, 255, 255)] * num_colors
    
    # Save to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    # Extract colors
    color_thief = ColorThief(img_byte_arr)
    palette = color_thief.get_palette(color_count=num_colors, quality=1)
    
    return palette


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color code"""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def pil_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string for display
    
    Args:
        image: PIL Image
    
    Returns:
        Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def enhance_prompt_with_style(base_prompt: str, style: str = "modern") -> str:
    """
    Enhance user prompt with style keywords
    
    Args:
        base_prompt: User's text description
        style: Style preset (modern, vintage, minimal, playful, corporate)
    
    Returns:
        Enhanced prompt string
    """
    style_keywords = {
        "modern": "sleek, contemporary, clean lines, geometric",
        "vintage": "retro, classic, nostalgic, hand-drawn style",
        "minimal": "minimalist, simple, elegant, negative space",
        "playful": "fun, colorful, friendly, rounded shapes",
        "corporate": "professional, sophisticated, trustworthy, bold",
        "tech": "futuristic, digital, innovative, gradient",
        "organic": "natural, flowing, curved, handcrafted"
    }
    
    style_suffix = style_keywords.get(style.lower(), "")
    
    if style_suffix:
        return f"{base_prompt}, {style_suffix}"
    return base_prompt


def extract_keywords_from_text(text: str) -> dict:
    """
    Simple keyword extraction for colors and styles from user input
    
    Args:
        text: User input text
    
    Returns:
        Dictionary with detected keywords
    """
    text_lower = text.lower()
    
    # Color detection
    colors = ["red", "blue", "green", "yellow", "orange", "purple", 
              "pink", "black", "white", "gray", "gold", "silver"]
    detected_colors = [c for c in colors if c in text_lower]
    
    # Style detection
    styles = ["modern", "vintage", "minimal", "playful", "corporate", 
              "tech", "organic", "elegant", "bold", "simple"]
    detected_styles = [s for s in styles if s in text_lower]
    
    return {
        "colors": detected_colors,
        "styles": detected_styles
    }


def create_image_grid(images: List[Image.Image], cols: int = 2) -> Image.Image:
    """
    Create a grid layout of multiple images
    
    Args:
        images: List of PIL Images
        cols: Number of columns in grid
    
    Returns:
        Combined grid image
    """
    if not images:
        return None
    
    # Calculate grid dimensions
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    # Get dimensions from first image
    img_width, img_height = images[0].size
    
    # Create blank canvas
    grid_width = cols * img_width
    grid_height = rows * img_height
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * img_width
        y = row * img_height
        grid.paste(img, (x, y))
    
    return grid


def apply_logo_mockup(logo: Image.Image, mockup_type: str = "white") -> Image.Image:
    """
    Apply simple mockup background to logo
    
    Args:
        logo: PIL Image of logo
        mockup_type: Type of mockup (white, dark, gradient)
    
    Returns:
        Logo with mockup background
    """
    # Create canvas slightly larger than logo
    canvas_size = (logo.width + 100, logo.height + 100)
    
    if mockup_type == "dark":
        canvas = Image.new('RGB', canvas_size, color=(30, 30, 30))
    elif mockup_type == "gradient":
        # Create simple gradient
        canvas = Image.new('RGB', canvas_size, color=(240, 240, 240))
    else:  # white
        canvas = Image.new('RGB', canvas_size, color=(255, 255, 255))
    
    # Center the logo
    x = (canvas.width - logo.width) // 2
    y = (canvas.height - logo.height) // 2
    
    # Paste logo (handle transparency)
    if logo.mode == 'RGBA':
        canvas.paste(logo, (x, y), logo)
    else:
        canvas.paste(logo, (x, y))
    
    return canvas