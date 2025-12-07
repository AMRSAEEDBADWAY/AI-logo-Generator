"""
AI Logo Generator & Classifier - Streamlit Web App
Complete web interface for generating and classifying logos
"""

import streamlit as st
from PIL import Image
import numpy as np
import json
from pathlib import Path
import io

# PyTorch for classifier
import torch
import torch.nn as nn
from torchvision import models, transforms

# Import custom modules
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import CLASSIFIER_CONFIG, GENERATION_CONFIG, UI_CONFIG, MODELS_DIR
from src.utils import (
    remove_background,
    extract_dominant_colors,
    rgb_to_hex,
    enhance_prompt_with_style,
    apply_logo_mockup
)
from src.generate_logo import LogoGenerator

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title=UI_CONFIG['page_title'],
    page_icon=UI_CONFIG['page_icon'],
    layout=UI_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    /* Dark Professional Theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%);
    }
    
    /* Main Headers */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.3);
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #a0a0ff;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Connection Status Bar */
    .connection-bar {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .connection-status {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    .status-dot.connected {
        background: #00ff88;
        box-shadow: 0 0 10px #00ff88;
    }
    
    .status-dot.disconnected {
        background: #ff4444;
        box-shadow: 0 0 10px #ff4444;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Logo Gallery */
    .logo-gallery {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }
    
    .logo-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        transition: transform 0.3s ease;
    }
    
    .logo-item:hover {
        transform: scale(1.02);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Success/Warning/Error */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1) !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
    }
    
    .stWarning {
        background: rgba(255, 170, 0, 0.1) !important;
        border: 1px solid rgba(255, 170, 0, 0.3) !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE INITIALIZATION ==========
if 'generated_images' not in st.session_state:
    st.session_state.generated_images = []
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'classifier_model' not in st.session_state:
    st.session_state.classifier_model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False


# ========== LOAD MODELS ==========
@st.cache_resource
def load_classifier():
    """Load the trained CNN classifier (PyTorch)"""
    model_path = CLASSIFIER_CONFIG['model_path']
    
    if not model_path.exists():
        return None, None
    
    try:
        # Load class names
        class_indices_path = MODELS_DIR / "class_indices.json"
        if class_indices_path.exists():
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
                class_names = {v: k for k, v in class_indices.items()}
        else:
            class_names = {i: name for i, name in enumerate(CLASSIFIER_CONFIG['classes'])}
            
        num_classes = len(class_names)

        # Recreate the model architecture (EfficientNet-B0)
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, num_classes)
        
        # Load state dict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None, None


@st.cache_resource
def load_generator(api_url: str = None):
    """Load the logo generator (connects to Colab API)"""
    try:
        generator = LogoGenerator(api_url=api_url)
        return generator
    except Exception as e:
        st.error(f"Error initializing generator: {e}")
        return None

# ========== SIDEBAR ==========
def render_sidebar():
    """Render sidebar with navigation and settings"""
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.8rem; margin: 0;'>AI Logo Studio</h1>
            <p style='color: #888; font-size: 0.8rem; margin-top: 0.5rem;'>Powered by SDXL + LogoRedmond</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("##### 📍 Navigation")
        page = st.radio(
            "Navigation",
            ["🎨 Generate Logo", "🔍 Classify Logo", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Settings (only for Generate page)
        if page == "🎨 Generate Logo":
            st.markdown("##### 🎨 Style")
            st.session_state.style_preset = st.selectbox(
                "Style Preset",
                ["modern", "minimal", "vintage", "playful", "corporate", "tech", "organic"],
                label_visibility="collapsed"
            )
            
            st.markdown("##### �️ Quantity")
            st.session_state.num_images = st.slider(
                "Number of Images",
                min_value=1,
                max_value=4,
                value=2,
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                st.session_state.num_steps = st.slider(
                    "Quality (Steps)",
                    min_value=15,
                    max_value=40,
                    value=25
                )
                
                st.session_state.lora_scale = st.slider(
                    "Logo Style Strength",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.9,
                    step=0.1
                )
                
                st.session_state.guidance_scale = st.slider(
                    "Prompt Adherence",
                    min_value=5.0,
                    max_value=12.0,
                    value=7.5,
                    step=0.5
                )
        
        st.markdown("---")
        st.markdown("<p style='text-align: center; color: #555; font-size: 0.7rem;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
        
        return page

# ========== PAGE: GENERATE LOGO ==========
def page_generate_logo():
    """Logo generation page with professional UI"""
    
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Logo Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Create stunning professional logos with SDXL + LogoRedmond AI</p>', unsafe_allow_html=True)
    
    # ===== CONNECTION SECTION =====
    st.markdown("### 🔌 Colab Connection")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        api_url = st.text_input(
            "Colab API URL",
            value=st.session_state.get("colab_api_url", ""),
            placeholder="Paste your ngrok URL here (e.g., https://xxxx.ngrok-free.dev)",
            label_visibility="collapsed"
        )
    
    with col2:
        connect_btn = st.button("🔗 Connect", use_container_width=True)
    
    with col3:
        if st.session_state.get("api_connected", False):
            st.markdown("<div style='background: rgba(0,255,136,0.2); padding: 0.5rem 1rem; border-radius: 8px; text-align: center;'>✅ Connected</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='background: rgba(255,68,68,0.2); padding: 0.5rem 1rem; border-radius: 8px; text-align: center;'>⚠️ Disconnected</div>", unsafe_allow_html=True)
    
    # Handle connection
    if connect_btn and api_url:
        st.session_state["colab_api_url"] = api_url
        try:
            st.session_state.generator = LogoGenerator(api_url=api_url)
            if st.session_state.generator.is_remote:
                st.session_state.api_connected = True
                st.success("✅ Successfully connected to SDXL + LogoRedmond on Colab!")
                st.rerun()
            else:
                st.session_state.api_connected = False
                st.error("❌ Could not connect. Check if Colab is running.")
        except Exception as e:
            st.session_state.api_connected = False
            st.error(f"❌ Connection failed: {e}")
    elif api_url and not connect_btn:
        st.session_state["colab_api_url"] = api_url
    
    # Show instructions if not connected
    if not st.session_state.get("api_connected", False):
        with st.expander("📖 How to Connect?", expanded=True):
            st.markdown("""
            **Quick Setup:**
            1. Open `SDXL_LogoRedmond_API.ipynb` on [Google Colab](https://colab.research.google.com)
            2. Run all cells (Ctrl+F9)
            3. Copy the ngrok URL that appears
            4. Paste it above and click **Connect**
            
            *The URL looks like: `https://xxxx-xxx.ngrok-free.dev`*
            """)
        return  # Stop here if not connected
    
    st.markdown("---")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Describe your logo",
            placeholder="e.g., Modern coffee shop logo with minimalist coffee cup, warm colors",
            height=100,
            help="Be specific about what you want to see"
        )
    
    with col2:
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific about your business/brand
        - Mention preferred colors
        - Describe the style (modern, vintage, etc.)
        - Add industry context
        """)
    
    # Advanced options
    with st.expander("🎛️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            custom_suffix = st.text_input(
                "Custom Prompt Suffix",
                value="",
                help="Add additional keywords to enhance the prompt"
            )
        
        with col2:
            negative_prompt = st.text_area(
                "Negative Prompt",
                value=GENERATION_CONFIG['negative_prompt'],
                help="Things you don't want in the logo",
                height=100
            )
    
    # Generate button
    if st.button("✨ Generate Logo", type="primary", disabled=(len(prompt.strip()) == 0)):
        if prompt.strip():
            # Build final prompt
            style = st.session_state.get('style_preset', 'modern')
            enhanced_prompt = enhance_prompt_with_style(prompt, style)
            
            if custom_suffix:
                enhanced_prompt += f", {custom_suffix}"
            
            st.session_state.current_prompt = enhanced_prompt
            
            # Show prompt
            with st.expander("📝 Full Prompt Used"):
                st.code(enhanced_prompt, language="text")
            
            # Generate images
            with st.spinner(f"🎨 Generating {st.session_state.num_images} logo(s) via Colab... (30-60s each)"):
                try:
                    seed = st.session_state.seed if st.session_state.get('use_seed', False) else None
                    
                    images = st.session_state.generator.generate(
                        prompt=prompt,
                        style=style,
                        num_images=st.session_state.get('num_images', 2),
                        num_steps=st.session_state.get('num_steps', 25),
                        guidance_scale=st.session_state.get('guidance_scale', 7.5),
                        lora_scale=st.session_state.get('lora_scale', 0.9),
                        seed=seed
                    )
                    
                    st.session_state.generated_images = images
                    st.success(f"✅ Generated {len(images)} logo(s) successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during generation: {e}")
                    st.info("💡 Check if Colab is still running and the URL is correct.")
    
    # Display generated images
    if st.session_state.generated_images:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Generated Logos</h2>', unsafe_allow_html=True)
        
        # Create grid
        cols = st.columns(2)
        
        for idx, img in enumerate(st.session_state.generated_images):
            with cols[idx % 2]:
                st.image(img, use_container_width=True)
                
                # Download buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download original
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    st.download_button(
                        label="📥 PNG",
                        data=buf.getvalue(),
                        file_name=f"logo_{idx+1}.png",
                        mime="image/png",
                        key=f"download_png_{idx}"
                    )
                
                with col2:
                    # Remove background
                    if st.button("🎭 Remove BG", key=f"remove_bg_{idx}"):
                        with st.spinner("Removing background..."):
                            try:
                                img_no_bg = remove_background(img)
                                st.image(img_no_bg, caption="Background Removed")
                                
                                buf = io.BytesIO()
                                img_no_bg.save(buf, format='PNG')
                                st.download_button(
                                    label="📥 Download",
                                    data=buf.getvalue(),
                                    file_name=f"logo_{idx+1}_nobg.png",
                                    mime="image/png",
                                    key=f"download_nobg_{idx}"
                                )
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                with col3:
                    # Show colors
                    if st.button("🎨 Colors", key=f"colors_{idx}"):
                        with st.spinner("Extracting colors..."):
                            try:
                                colors = extract_dominant_colors(img, num_colors=5)
                                st.write("Dominant Colors:")
                                color_html = ""
                                for color in colors:
                                    hex_color = rgb_to_hex(color)
                                    color_html += f'<div style="display:inline-block; width:40px; height:40px; background-color:{hex_color}; margin:2px; border:1px solid #ccc;"></div>'
                                st.markdown(color_html, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                st.markdown("---")


# ========== PAGE: CLASSIFY LOGO ==========
def page_classify_logo():
    """Logo classification page"""
    st.markdown('<h1 class="main-header">🔍 Logo Classifier</h1>', unsafe_allow_html=True)
    st.markdown("Upload a logo image and let AI predict its category!")
    
    # Load classifier
    if st.session_state.classifier_model is None:
        with st.spinner("Loading classifier model..."):
            st.session_state.classifier_model, st.session_state.class_names = load_classifier()
    
    if st.session_state.classifier_model is None:
        st.warning("⚠️ Classifier model not found. Please train the model first.")
        st.info("Run: `python src/train_classifier.py`")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Logo Image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a logo image for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Logo", use_container_width=True)
        
        with col2:
            st.markdown("### Image Information")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} px")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        # Classify button
        if st.button("🔍 Classify Logo", type="primary"):
            with st.spinner("Analyzing logo..."):
                try:
                    # Preprocess image for PyTorch
                    transform = transforms.Compose([
                        transforms.Resize(CLASSIFIER_CONFIG['image_size']),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        img_rgb = image.convert('RGB')
                    else:
                        img_rgb = image
                        
                    input_tensor = transform(img_rgb).unsqueeze(0)
                    
                    # Move to device
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    input_tensor = input_tensor.to(device)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = st.session_state.classifier_model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        
                    # Get results
                    probs_np = probabilities.cpu().numpy()[0]
                    predicted_class_idx = np.argmax(probs_np)
                    confidence = probs_np[predicted_class_idx]
                    
                    # Display results
                    st.markdown("---")
                    st.markdown('<h2 class="sub-header">Classification Results</h2>', unsafe_allow_html=True)
                    
                    # Main prediction
                    predicted_class = st.session_state.class_names[predicted_class_idx]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Category", predicted_class)
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # All predictions
                    st.markdown("### All Predictions")
                    
                    # Sort by confidence
                    sorted_indices = np.argsort(probs_np)[::-1]
                    
                    for idx in sorted_indices:
                        class_name = st.session_state.class_names[idx]
                        conf = probs_np[idx]
                        
                        st.progress(float(conf), text=f"{class_name}: {conf*100:.2f}%")
                
                except Exception as e:
                    st.error(f"❌ Error during classification: {e}")


# ========== PAGE: ABOUT ==========
def page_about():
    """About page with model information"""
    st.markdown('<h1 class="main-header">ℹ️ About This Project</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Project Overview
    
    This is a complete AI-powered logo generation and classification system that combines:
    
    1. **Deep Learning Classifier** - Custom CNN trained to categorize logos
    2. **AI Logo Generator** - SDXL/FLUX model for creating professional logos
    3. **Streamlit Web App** - User-friendly interface for both features
    
    ---
    
    ## 🧠 Model Architecture
    
    ### Logo Classifier (CNN)
    - **Architecture:** EfficientNet-B0 (PyTorch)
    - **Input:** 224x224 RGB images
    - **Output:** Multi-class classification
    - **Training:** Custom dataset with data augmentation
    
    ### Logo Generator
    - **Model:** Stable Diffusion XL (SDXL) or FLUX
    - **Technique:** Text-to-image diffusion
    - **Prompt Engineering:** Optimized for logo generation
    - **Optional:** LoRA fine-tuning support
    
    ---
    
    ## 🚀 Features
    
    ### Generation
    - ✨ AI-powered logo creation from text
    - 🎨 Multiple style presets
    - 🔧 Advanced prompt customization
    - 📥 Download as PNG
    - 🎭 Background removal
    - 🌈 Color palette extraction
    
    ### Classification
    - 🔍 Automatic logo category detection
    - 📊 Confidence scores for all categories
    - 🎯 Trained on diverse logo dataset
    
    ---
    
    ## 📚 Technologies Used
    
    - **Deep Learning:** PyTorch
    - **Image Generation:** Diffusers, Stable Diffusion XL
    - **Web Framework:** Streamlit
    - **Image Processing:** PIL, OpenCV, rembg
    - **Python:** 3.8+
    
    ---
    
    ## 📖 How to Use
    
    ### Generate a Logo:
    1. Navigate to "Generate Logo" page
    2. Describe your desired logo
    3. Choose a style preset
    4. Click "Generate Logo"
    5. Download your favorites!
    
    ### Classify a Logo:
    1. Navigate to "Classify Logo" page
    2. Upload a logo image
    3. Click "Classify Logo"
    4. View predictions and confidence scores
    
    ---
    
    ## 🛠️ Setup Instructions
    
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt
    
    # 2. Set up HuggingFace token (for SDXL/FLUX)
    export HUGGINGFACE_TOKEN="your_token_here"
    
    # 3. Train classifier (optional)
    python src/train_classifier.py
    
    # 4. Run the app
    streamlit run app.py
    ```
    
    ---
    
    ## 📧 Support
    
    For issues or questions, please check the README.md file or documentation.
    
    ---
    
    **Made with ❤️ by AI enthusiasts**
    """)
    
    # Model status
    st.markdown("---")
    st.markdown("### 🔧 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        classifier_status = "✅ Loaded" if st.session_state.classifier_model else "❌ Not Loaded"
        st.info(f"**Classifier:** {classifier_status}")
    
    with col2:
        generator_status = "✅ Loaded" if st.session_state.generator else "❌ Not Loaded"
        st.info(f"**Generator:** {generator_status}")


# ========== MAIN APP ==========
def main():
    """Main application logic"""
    
    # Render sidebar and get current page
    page = render_sidebar()
    
    # Route to appropriate page
    if page == "🎨 Generate Logo":
        page_generate_logo()
    elif page == "🔍 Classify Logo":
        page_classify_logo()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()