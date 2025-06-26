import requests
import unittest
import os
import io
from datetime import datetime

class ResumeAnalyzerAPITest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ResumeAnalyzerAPITest, self).__init__(*args, **kwargs)
        self.base_url = "http://localhost:8001"
        self.resume_id = None
        
    def test_01_health_check(self):
        """Test the health check endpoint"""
        print("\n🔍 Testing health check endpoint...")
        response = requests.get(f"{self.base_url}/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        print("✅ Health check passed")
        
    def test_02_upload_resume(self):
        """Test resume upload functionality"""
        print("\n🔍 Testing resume upload...")
        
        # Create a simple text file for testing
        test_resume = """
        John Doe
        Software Engineer
        
        SKILLS
        Python, JavaScript, React, Node.js, SQL
        
        EXPERIENCE
        Senior Developer - ABC Company
        2018-2022
        - Developed web applications using React and Node.js
        - Managed database using SQL
        
        EDUCATION
        Bachelor of Computer Science
        XYZ University, 2018
        """
        
        files = {'file': ('test_resume.txt', test_resume)}
        response = requests.post(f"{self.base_url}/api/upload-resume", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("resume_id", data)
        self.assertIn("preview", data)
        
        # Save resume_id for later tests
        self.resume_id = data["resume_id"]
        print(f"✅ Resume upload passed - ID: {self.resume_id}")
        
    def test_03_analyze_resume(self):
        """Test resume analysis functionality"""
        print("\n🔍 Testing resume analysis...")
        
        test_resume = """
        John Doe
        Software Engineer
        
        SKILLS
        Python, JavaScript, React, Node.js, SQL
        
        EXPERIENCE
        Senior Developer - ABC Company
        2018-2022
        - Developed web applications using React and Node.js
        - Managed database using SQL
        
        EDUCATION
        Bachelor of Computer Science
        XYZ University, 2018
        """
        
        job_description = """
        We are looking for a Software Engineer with experience in:
        - Python
        - JavaScript
        - React
        - Node.js
        - MongoDB
        
        Responsibilities:
        - Develop web applications
        - Work with databases
        - Collaborate with team members
        """
        
        payload = {
            "resume_text": test_resume,
            "job_description": job_description
        }
        
        response = requests.post(f"{self.base_url}/api/analyze-resume", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn("resume_id", data)
        self.assertIn("match_percentage", data)
        self.assertIn("skills_analysis", data)
        self.assertIn("resume_score", data)
        
        # Verify skills analysis
        skills = data["skills_analysis"]
        self.assertIn("detected_skills", skills)
        self.assertIn("matched_skills", skills)
        self.assertIn("missing_skills", skills)
        
        # Verify resume score
        score = data["resume_score"]
        self.assertIn("overall_score", score)
        self.assertIn("ats_score", score)
        self.assertIn("clarity_score", score)
        self.assertIn("relevance_score", score)
        
        print("✅ Resume analysis passed")
        
    def test_04_analyze_without_job_description(self):
        """Test resume analysis without job description"""
        print("\n🔍 Testing resume analysis without job description...")
        
        test_resume = """
        John Doe
        Software Engineer
        
        SKILLS
        Python, JavaScript, React, Node.js, SQL
        
        EXPERIENCE
        Senior Developer - ABC Company
        2018-2022
        - Developed web applications using React and Node.js
        - Managed database using SQL
        
        EDUCATION
        Bachelor of Computer Science
        XYZ University, 2018
        """
        
        payload = {
            "resume_text": test_resume,
            "job_description": None
        }
        
        response = requests.post(f"{self.base_url}/api/analyze-resume", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn("resume_id", data)
        self.assertIn("match_percentage", data)
        self.assertIn("skills_analysis", data)
        self.assertIn("resume_score", data)
        
        # Match percentage should be 0 without job description
        self.assertEqual(data["match_percentage"], 0)
        
        print("✅ Resume analysis without job description passed")
        
    def test_05_get_resume_by_id(self):
        """Test getting resume by ID"""
        if not self.resume_id:
            self.skipTest("No resume ID available from previous tests")
            
        print(f"\n🔍 Testing get resume by ID: {self.resume_id}...")
        
        response = requests.get(f"{self.base_url}/api/resume/{self.resume_id}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["id"], self.resume_id)
        self.assertIn("resume_text", data)
        self.assertIn("filename", data)
        
        print("✅ Get resume by ID passed")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)