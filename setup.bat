#!/bin/bash
# Quick Setup Script for AI Logo Generator
# For Linux/Mac users

echo "=========================================="
echo "🎨 AI Logo Generator - Quick Setup"
echo "=========================================="

# Check Python version
echo ""
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "   Found: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
echo "   ✅ Virtual environment created"

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip
echo "   ✅ Pip upgraded"

# Install requirements
echo ""
echo "📚 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt
echo "   ✅ Dependencies installed"

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/raw/train
mkdir -p data/raw/test
mkdir -p data/processed
mkdir -p models/lora
mkdir -p notebooks
echo "   ✅ Directories created"

# Create .env file template
echo ""
echo "🔐 Creating .env template..."
cat > .env << EOF
# HuggingFace Token (required for SDXL/FLUX)
HUGGINGFACE_TOKEN=your_token_here

# Optional: Model selection
# MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
# MODEL_ID=black-forest-labs/FLUX.1-schnell
EOF
echo "   ✅ .env template created"

# Create __init__.py files
echo ""
echo "📝 Creating Python package files..."
touch src/__init__.py
echo "   ✅ Package files created"

# Summary
echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Set your HuggingFace token:"
echo "   • Edit .env file"
echo "   • Or: export HUGGINGFACE_TOKEN='your_token'"
echo ""
echo "2. Accept model licenses:"
echo "   • Visit: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
echo "   • Click 'Agree and access repository'"
echo ""
echo "3. Prepare dataset (optional):"
echo "   • Place training images in data/raw/train/"
echo "   • Organize by category folders"
echo ""
echo "4. Train classifier (optional):"
echo "   • Run: python src/train_classifier.py"
echo ""
echo "5. Start the app:"
echo "   • Run: streamlit run app.py"
echo ""
echo "=========================================="
echo "Happy Logo Creating! 🎨✨"
echo "=========================================="


# ========== WINDOWS VERSION (save as setup.bat) ==========
: '
@echo off
echo ==========================================
echo 🎨 AI Logo Generator - Quick Setup
echo ==========================================

echo.
echo 📌 Checking Python version...
python --version

echo.
echo 📦 Creating virtual environment...
python -m venv venv
echo    ✅ Virtual environment created

echo.
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat
echo    ✅ Virtual environment activated

echo.
echo ⬆️  Upgrading pip...
pip install --upgrade pip
echo    ✅ Pip upgraded

echo.
echo 📚 Installing dependencies...
pip install -r requirements.txt
echo    ✅ Dependencies installed

echo.
echo 📁 Creating directory structure...
mkdir data\raw\train 2>nul
mkdir data\raw\test 2>nul
mkdir data\processed 2>nul
mkdir models\lora 2>nul
mkdir notebooks 2>nul
echo    ✅ Directories created

echo.
echo 🔐 Creating .env template...
(
echo # HuggingFace Token
echo HUGGINGFACE_TOKEN=your_token_here
) > .env
echo    ✅ .env template created

echo.
echo ==========================================
echo ✅ Setup Complete!
echo ==========================================
echo.
echo Next Steps:
echo 1. Edit .env file with your HuggingFace token
echo 2. Accept SDXL license on HuggingFace
echo 3. Prepare dataset in data/raw/train/
echo 4. Train classifier: python src/train_classifier.py
echo 5. Start app: streamlit run app.py
echo.
echo ==========================================
echo Happy Logo Creating! 🎨✨
echo ==========================================

pause
'