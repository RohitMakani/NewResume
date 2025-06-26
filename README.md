# 🚀 AI Resume Analyzer

A comprehensive resume analysis tool powered by AI that provides detailed feedback, skills gap analysis, and improvement suggestions.

## ✨ Features

- **📄 Multi-format Resume Upload**: Supports PDF, DOCX, TXT, and image files (JPG/PNG) with OCR
- **🎯 Skills Analysis**: Extracts and analyzes technical and soft skills
- **📊 Resume Scoring**: Comprehensive scoring (ATS, Clarity, Relevance, Completeness)
- **🤖 AI-Powered Suggestions**: Get targeted improvement recommendations using Groq LLM
- **💼 Job Matching**: Compare resume against job descriptions with match percentage
- **🎯 Role Recommendations**: Get suitable job role suggestions based on your profile
- **✨ Resume Optimization**: AI-generated improved resume versions

## 🛠️ Tech Stack

- **Backend**: FastAPI + SQLAlchemy + SQLite
- **Frontend**: Vanilla HTML/CSS/JavaScript  
- **AI/ML**: Groq LLM (LLaMA 3), spaCy, Sentence Transformers
- **Document Processing**: pdfplumber, python-docx, pytesseract, OpenCV

## 📋 Requirements

- Python 3.8+
- Groq API Key (for AI features)
- Tesseract OCR (for image processing)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Navigate to project directory
cd ai-resume-analyzer

# Install dependencies
chmod +x scripts/install_dependencies.sh
./scripts/install_dependencies.sh
```

### 2. Configure Environment
```bash
# Edit backend/.env file with your API key
nano backend/.env

# Add your Groq API key:
GROQ_API_KEY=your_groq_api_key_here
DATABASE_URL=sqlite:///./resume_analyzer.db
```

### 3. Run the Application

#### Option A: Using Python directly
```bash
cd backend
python server.py
```

#### Option B: Using Supervisor (Recommended)
```bash
# Start with supervisor
sudo supervisord -c supervisord.conf

# Check status
sudo supervisorctl status

# Restart if needed
sudo supervisorctl restart backend
```

### 4. Access the Application
Open your browser and go to: **http://localhost:8001**

## 🔧 Local Development Setup

### Manual Installation Steps

1. **Install Python Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install spaCy Language Model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Install Tesseract OCR**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-eng
   
   # macOS
   brew install tesseract
   
   # Windows
   # Download from: https://github.com/UB-Mannheim/tesseract/wiki
   ```

4. **Set Environment Variables**
   ```bash
   export GROQ_API_KEY="your_api_key_here"
   export DATABASE_URL="sqlite:///./resume_analyzer.db"
   ```

5. **Run the Server**
   ```bash
   cd backend
   python server.py
   ```

## 📊 API Endpoints

- `POST /api/upload-resume` - Upload and parse resume file
- `POST /api/analyze-resume` - Perform comprehensive resume analysis  
- `GET /api/resume/{resume_id}` - Get resume by ID
- `GET /api/analysis/{analysis_id}` - Get analysis by ID
- `GET /api/health` - Health check endpoint

## 🎯 Usage Guide

1. **Upload Resume**: Select your resume file (PDF, DOCX, TXT, or image)
2. **Add Job Description** (Optional): Paste job description for targeted analysis
3. **Click Analyze**: Get comprehensive analysis including:
   - Overall resume score and individual metrics
   - Skills analysis (detected, matched, missing)
   - AI-powered improvement suggestions
   - Job role recommendations
   - Optimized resume version

## 🔍 Supported File Formats

- **PDF**: Text extraction with OCR fallback for scanned documents
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files
- **Images**: JPG, PNG with OCR text extraction

## 🤖 AI Features

The application uses Groq's LLaMA 3 model for:
- Generating targeted improvement suggestions
- Creating optimized resume versions
- Providing job role recommendations

*Note: AI features require a valid Groq API key*

## 🗄️ Database

Uses SQLite database with two main tables:
- `resumes`: Stores uploaded resume data
- `analyses`: Stores analysis results and history

## 🛡️ Error Handling

- Comprehensive error handling for file processing
- Graceful degradation when AI services are unavailable
- User-friendly error messages in the UI

## 📝 Logs

Application logs are available at:
- `/var/log/supervisor/backend.out.log` (stdout)
- `/var/log/supervisor/backend.err.log` (stderr)

## 🚨 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Tesseract Not Found**
   ```bash
   # Check if installed
   tesseract --version
   
   # Install if missing (Ubuntu)
   sudo apt-get install tesseract-ocr
   ```

3. **Permission Errors**
   ```bash
   # Fix file permissions
   chmod +x scripts/install_dependencies.sh
   ```

4. **API Key Issues**
   - Verify GROQ_API_KEY is set correctly in backend/.env
   - Check API key validity at groq.com

### Performance Tips

- Large PDF files may take longer to process
- Image files require OCR processing which can be slower
- Job description analysis improves with longer, detailed descriptions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Groq for LLM API
- spaCy for NLP processing
- Sentence Transformers for semantic analysis
- FastAPI for the robust backend framework