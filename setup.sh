#!/bin/bash

# Setup script for Mechanistic Interpretability Workshop
# HasanLabs - August 2025

echo "================================================"
echo "  Mechanistic Interpretability Workshop Setup  "
echo "                  HasanLabs                    "
echo "================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo "✅ Python $python_version found"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment created"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."

cd steering-demo
pip install -r requirements.txt

echo "✅ Dependencies installed"

# Check for Hugging Face login
echo ""
echo "Checking Hugging Face access..."

if ! python3 -c "from huggingface_hub import HfFolder; token = HfFolder.get_token(); exit(0 if token else 1)" 2>/dev/null; then
    echo ""
    echo "⚠️  You need to log in to Hugging Face to access Llama models"
    echo "Run: huggingface-cli login"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo ""
    read -p "Do you want to login now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    fi
fi

# Create vectors directory
mkdir -p llama_3b_steered/vectors

echo ""
echo "================================================"
echo "              Setup Complete! 🎉               "
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Generate steering vectors (takes ~10 minutes):"
echo "   python generate_vectors.py"
echo ""
echo "2. Test the system:"
echo "   python test_steering.py"
echo ""
echo "3. Run the web demo:"
echo "   cd steering-demo"
echo "   python web_interface/app.py"
echo ""
echo "4. Open browser to:"
echo "   http://localhost:5050"
echo ""
echo "For the workshop slides:"
echo "   cd slides && npm install && npm run dev"
echo ""