import os
import uuid
import base64
import io
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
import uvicorn

# Document processing imports
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import cv2
import numpy as np

# NLP imports
from sentence_transformers import SentenceTransformer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Groq integration
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="AI Resume Analyzer", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

# Initialize clients
client = AsyncIOMotorClient(MONGO_URL)
db = client.resume_analyzer
resumes_collection = db.resumes
analyses_collection = db.analyses

# Initialize ML models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    logger.warning("spaCy model not found. Some features may be limited.")
    nlp = None

# Initialize Groq client
groq_client = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    logger.warning("GROQ_API_KEY not found. AI features will be disabled.")


# Pydantic models
class ResumeAnalysisRequest(BaseModel):
    resume_text: str
    job_description: Optional[str] = None
    target_role: Optional[str] = None


class SkillsAnalysis(BaseModel):
    detected_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    skill_match_percentage: float


class ResumeScore(BaseModel):
    overall_score: float
    ats_score: float
    clarity_score: float
    relevance_score: float
    completeness_score: float


class EnhancementSuggestion(BaseModel):
    section: str
    current_text: str
    suggested_improvement: str
    reasoning: str


class ResumeAnalysisResponse(BaseModel):
    resume_id: str
    match_percentage: float
    skills_analysis: SkillsAnalysis
    resume_score: ResumeScore
    enhancement_suggestions: List[EnhancementSuggestion]
    ai_generated_resume: Optional[str] = None
    job_role_recommendations: List[str]


# Utility functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF using pdfplumber with OCR fallback."""
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\
"
                else:
                    # OCR fallback for scanned PDFs
                    try:
                        img = page.to_image()
                        pil_img = img.original
                        ocr_text = pytesseract.image_to_string(pil_img)
                        text += ocr_text + "\
"
                    except Exception as e:
                        logger.warning(f"OCR failed for page: {e}")
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        doc = Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\
"
        return text.strip()
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from DOCX")


def extract_text_from_image(file_content: bytes) -> str:
    """Extract text from image using OCR."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess image for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Extract text
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        logger.error(f"Image OCR failed: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from image")


def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from resume text using NLP."""
    # Common skills database (simplified)
    common_skills = [
        # Programming Languages
        "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "Go", "Rust", "PHP", "Swift",
        "Kotlin", "TypeScript", "Scala", "R", "MATLAB", "SQL", "HTML", "CSS",

        # Frameworks & Libraries
        "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring", "Express",
        "React Native", "Flutter", "Laravel", "Rails", "ASP.NET", "jQuery",

        # Databases
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch", "SQLite", "Oracle",
        "Cassandra", "DynamoDB", "Firebase",

        # Cloud & DevOps
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Jenkins", "Git", "GitHub",
        "GitLab", "CI/CD", "Terraform", "Ansible", "Linux", "Unix",

        # Data Science & AI
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Pandas", "NumPy",
        "Scikit-learn", "Jupyter", "Apache Spark", "Hadoop", "Power BI", "Tableau",

        # Soft Skills
        "Leadership", "Communication", "Project Management", "Team Management", "Problem Solving",
        "Critical Thinking", "Agile", "Scrum", "Product Management"
    ]

    detected_skills = []
    text_lower = text.lower()

    for skill in common_skills:
        if skill.lower() in text_lower:
            detected_skills.append(skill)

    # Use spaCy for additional entity extraction if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT", "LANGUAGE"]:
                detected_skills.append(ent.text)

    return list(set(detected_skills))


def calculate_match_percentage(resume_text: str, job_description: str) -> float:
    """Calculate match percentage between resume and job description."""
    if not job_description:
        return 0.0

    try:
        # Semantic similarity using sentence transformers
        resume_embedding = sentence_model.encode([resume_text])
        job_embedding = sentence_model.encode([job_description])
        semantic_similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]

        # TF-IDF similarity
        tfidf_matrix = tfidf_vectorizer.fit_transform([resume_text, job_description])
        tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Combined score (weighted average)
        match_percentage = (semantic_similarity * 0.7 + tfidf_similarity * 0.3) * 100
        return min(match_percentage, 100.0)
    except Exception as e:
        logger.error(f"Match calculation failed: {e}")
        return 0.0


def analyze_skills_gap(resume_skills: List[str], job_description: str) -> SkillsAnalysis:
    """Analyze skills gap between resume and job requirements."""
    if not job_description:
        return SkillsAnalysis(
            detected_skills=resume_skills,
            matched_skills=[],
            missing_skills=[],
            skill_match_percentage=0.0
        )

    # Extract required skills from job description
    job_skills = extract_skills_from_text(job_description)

    # Find matched and missing skills
    matched_skills = list(set(resume_skills) & set(job_skills))
    missing_skills = list(set(job_skills) - set(resume_skills))

    # Calculate skill match percentage
    if job_skills:
        skill_match_percentage = (len(matched_skills) / len(job_skills)) * 100
    else:
        skill_match_percentage = 0.0

    return SkillsAnalysis(
        detected_skills=resume_skills,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        skill_match_percentage=skill_match_percentage
    )


def calculate_resume_score(resume_text: str, job_description: Optional[str] = None) -> ResumeScore:
    """Calculate comprehensive resume score."""
    # ATS Score (keyword density, formatting, etc.)
    ats_score = min(len(resume_text.split()) / 10, 100)  # Basic length-based scoring

    # Clarity Score (readability, structure)
    sentences = resume_text.split('.')
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    clarity_score = max(100 - (avg_sentence_length - 15) * 2, 0)  # Penalize very long sentences

    # Relevance Score (based on job description match)
    if job_description:
        relevance_score = calculate_match_percentage(resume_text, job_description)
    else:
        relevance_score = 50.0

    # Completeness Score (sections, contact info, etc.)
    sections = ["education", "experience", "skills", "contact"]
    completeness_score = sum(20 for section in sections if section in resume_text.lower()) + 20

    # Overall Score (weighted average)
    overall_score = (ats_score * 0.25 + clarity_score * 0.25 + relevance_score * 0.3 + completeness_score * 0.2)

    return ResumeScore(
        overall_score=min(overall_score, 100.0),
        ats_score=min(ats_score, 100.0),
        clarity_score=min(clarity_score, 100.0),
        relevance_score=min(relevance_score, 100.0),
        completeness_score=min(completeness_score, 100.0)
    )


async def generate_ai_suggestions(resume_text: str, job_description: Optional[str] = None) -> List[
    EnhancementSuggestion]:
    """Generate AI-powered enhancement suggestions using Groq."""
    if not groq_client:
        return []

    try:
        prompt = f"""
        Analyze this resume and provide specific improvement suggestions:

        Resume:
        {resume_text}

        {"Job Description: " + job_description if job_description else ""}

        Please provide 3-5 specific suggestions in the following format:
        Section: [section name]
        Current: [current text or issue]
        Suggested: [specific improvement]
        Reasoning: [why this improvement helps]

        Focus on:
        1. ATS optimization
        2. Impact quantification
        3. Keyword optimization
        4. Structure improvements
        5. Content enhancement
        """

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system",
                 "content": "You are an expert resume reviewer and career counselor with 10+ years of experience in HR and recruiting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2048
        )

        # Parse response into structured suggestions
        suggestions_text = response.choices[0].message.content
        suggestions = []

        # Simple parsing (can be improved with more sophisticated NLP)
        sections = suggestions_text.split("Section:")
        for section in sections[1:]:  # Skip first empty split
            lines = section.strip().split("\
")
            if len(lines) >= 4:
                section_name = lines[0].strip()
                current_text = lines[1].replace("Current:", "").strip() if "Current:" in lines[1] else ""
                suggested_text = lines[2].replace("Suggested:", "").strip() if "Suggested:" in lines[2] else ""
                reasoning = lines[3].replace("Reasoning:", "").strip() if "Reasoning:" in lines[3] else ""

                suggestions.append(EnhancementSuggestion(
                    section=section_name,
                    current_text=current_text,
                    suggested_improvement=suggested_text,
                    reasoning=reasoning
                ))

        return suggestions[:5]  # Limit to 5 suggestions
    except Exception as e:
        logger.error(f"AI suggestions generation failed: {e}")
        return []


async def generate_complete_resume(resume_text: str, job_description: Optional[str] = None) -> Optional[str]:
    """Generate a complete improved resume using Groq."""
    if not groq_client:
        return None

    try:
        prompt = f"""
        Create an improved, ATS-optimized version of this resume:

        Original Resume:
        {resume_text}

        {"Target Job Description: " + job_description if job_description else ""}

        Please create a complete, professional resume that:
        1. Is ATS-friendly with proper formatting
        2. Uses strong action verbs and quantified achievements
        3. Includes relevant keywords from the job description
        4. Has clear sections: Contact, Summary, Experience, Education, Skills
        5. Is tailored for the target role

        Return only the improved resume text, properly formatted.
        """

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system",
                 "content": "You are an expert resume writer who creates ATS-optimized, compelling resumes that get interviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=3000
        )

        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Complete resume generation failed: {e}")
        return None


async def get_job_role_recommendations(resume_text: str) -> List[str]:
    """Get job role recommendations based on resume content."""
    if not groq_client:
        return []

    try:
        prompt = f"""
        Based on this resume, suggest 5-7 suitable job roles/titles:

        Resume:
        {resume_text}

        Please provide job role recommendations that match the candidate's skills and experience.
        Return only a simple list of job titles, one per line.
        """

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system",
                 "content": "You are a career counselor who recommends suitable job roles based on candidate profiles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )

        roles = [role.strip() for role in response.choices[0].message.content.split("\
") if role.strip()]
        return roles[:7]  # Limit to 7 recommendations
    except Exception as e:
        logger.error(f"Job role recommendations failed: {e}")
        return []


# API Routes
@app.post("/api/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """Upload and parse resume file."""
    try:
        file_content = await file.read()

        # Determine file type and extract text
        if file.filename.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(file_content)
        elif file.filename.lower().endswith('.docx'):
            resume_text = extract_text_from_docx(file_content)
        elif file.filename.lower().endswith(('.txt', '.text')):
            resume_text = file_content.decode('utf-8')
        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            resume_text = extract_text_from_image(file_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")

        # Store resume in database
        resume_id = str(uuid.uuid4())
        resume_doc = {
            "_id": resume_id,
            "filename": file.filename,
            "resume_text": resume_text,
            "file_content": base64.b64encode(file_content).decode('utf-8'),
            "uploaded_at": datetime.utcnow()
        }

        await resumes_collection.insert_one(resume_doc)

        return {
            "resume_id": resume_id,
            "filename": file.filename,
            "text_length": len(resume_text),
            "preview": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
        }
    except Exception as e:
        logger.error(f"Resume upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-resume", response_model=ResumeAnalysisResponse)
async def analyze_resume(request: ResumeAnalysisRequest):
    """Perform comprehensive resume analysis."""
    try:
        resume_text = request.resume_text
        job_description = request.job_description

        # Extract skills from resume
        detected_skills = extract_skills_from_text(resume_text)

        # Analyze skills gap
        skills_analysis = analyze_skills_gap(detected_skills, job_description)

        # Calculate match percentage
        match_percentage = calculate_match_percentage(resume_text, job_description) if job_description else 0.0

        # Calculate resume score
        resume_score = calculate_resume_score(resume_text, job_description)

        # Generate AI suggestions
        enhancement_suggestions = await generate_ai_suggestions(resume_text, job_description)

        # Generate complete improved resume
        ai_generated_resume = await generate_complete_resume(resume_text, job_description)

        # Get job role recommendations
        job_role_recommendations = await get_job_role_recommendations(resume_text)

        # Create analysis result
        analysis_id = str(uuid.uuid4())
        analysis_result = ResumeAnalysisResponse(
            resume_id=analysis_id,
            match_percentage=match_percentage,
            skills_analysis=skills_analysis,
            resume_score=resume_score,
            enhancement_suggestions=enhancement_suggestions,
            ai_generated_resume=ai_generated_resume,
            job_role_recommendations=job_role_recommendations
        )

        # Store analysis in database
        analysis_doc = {
            "_id": analysis_id,
            "resume_text": resume_text,
            "job_description": job_description,
            "analysis_result": analysis_result.dict(),
            "analyzed_at": datetime.utcnow()
        }

        await analyses_collection.insert_one(analysis_doc)

        return analysis_result
    except Exception as e:
        logger.error(f"Resume analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/resume/{resume_id}")
async def get_resume(resume_id: str):
    """Get resume by ID."""
    try:
        resume = await resumes_collection.find_one({"_id": resume_id})
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")

        # Remove binary content for response
        resume.pop("file_content", None)
        return resume
    except Exception as e:
        logger.error(f"Get resume failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get analysis by ID."""
    try:
        analysis = await analyses_collection.find_one({"_id": analysis_id})
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return analysis
    except Exception as e:
        logger.error(f"Get analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "groq_available": groq_client is not None,
        "nlp_available": nlp is not None
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

