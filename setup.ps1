# Horizon AI POS System - Windows Setup Script
# ============================================

Write-Host "🏪 Setting up Horizon AI POS System for Windows..." -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python is not installed. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Check if pip is installed
try {
    $pipVersion = pip --version 2>&1
    Write-Host "✅ Found pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ pip is not installed. Please install pip first." -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "📥 Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "data" -Force | Out-Null
New-Item -ItemType Directory -Path "models" -Force | Out-Null
New-Item -ItemType Directory -Path "logs" -Force | Out-Null

# Setup initial data if needed
Write-Host "🤖 Setting up AI models..." -ForegroundColor Yellow
if (Test-Path "enhanced_setup.py") {
    python enhanced_setup.py
} else {
    Write-Host "⚠️ Enhanced setup script not found. You may need to train models manually." -ForegroundColor Yellow
}

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Run Streamlit app: streamlit run streamlit_app.py" -ForegroundColor White
Write-Host ""
Write-Host "🌐 The application will be available at: http://localhost:8501" -ForegroundColor Cyan