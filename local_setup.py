#!/usr/bin/env python3
"""
Local setup script for AI Resume Analyzer
Run this script to set up the application for local development
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version} is compatible")
    return True

def setup_project():
    """Main setup function"""
    print("🚀 Setting up AI Resume Analyzer...")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Download spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Downloading spaCy English model"):
        print("⚠️  spaCy model download failed. You may need to install it manually.")
    
    # Check for Tesseract
    try:
        subprocess.run(["tesseract", "--version"], check=True, capture_output=True)
        print("✅ Tesseract OCR is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Tesseract OCR not found. Install it for image processing:")
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   macOS: brew install tesseract")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    
    # Check environment file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating template...")
        with open(".env", "w") as f:
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
            f.write("DATABASE_URL=sqlite:///./resume_analyzer.db\n")
        print("📝 Please edit backend/.env and add your Groq API key")
    else:
        print("✅ Environment file exists")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Edit backend/.env and add your Groq API key")
    print("2. Run: cd backend && python server.py")
    print("3. Open browser to: http://localhost:8001")
    print("\n🔗 Get your Groq API key at: https://console.groq.com/")
    
    return True

if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)