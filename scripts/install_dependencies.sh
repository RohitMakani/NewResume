#!/bin/bash

echo "🚀 Installing Resume Analyzer Dependencies..."

# Install Python dependencies
echo "📦 Installing Python packages..."
cd /app/backend
pip install -r requirements.txt

# Download spaCy language model
echo "🧠 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Install Tesseract OCR (if not already installed)
echo "🔍 Checking Tesseract OCR..."
if ! command -v tesseract &> /dev/null; then
    echo "Installing Tesseract OCR..."
    apt-get update
    apt-get install -y tesseract-ocr tesseract-ocr-eng
else
    echo "Tesseract OCR already installed"
fi

# Create database directory
echo "💾 Setting up database..."
mkdir -p /app/backend/data

echo "✅ All dependencies installed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Set your GROQ_API_KEY in /app/backend/.env"
echo "2. Run the server: cd /app/backend && python server.py"
echo "3. Open browser to: http://localhost:8001"