#!/bin/bash
# Horizon AI POS System - Local Development Setup
# ===============================================

echo "🏪 Setting up Horizon AI POS System for local development..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models logs

# Setup initial data if needed
echo "🤖 Setting up AI models..."
if [ -f "enhanced_setup.py" ]; then
    python enhanced_setup.py
else
    echo "⚠️ Enhanced setup script not found. You may need to train models manually."
fi

echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run Streamlit app: streamlit run streamlit_app.py"
echo ""
echo "🌐 The application will be available at: http://localhost:8501"