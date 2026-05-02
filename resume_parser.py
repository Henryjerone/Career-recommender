"""
resume_parser.py — AI Career Recommendation System
Enhanced: Extracts ALL info from resume PDF
"""

import re

SKILL_KEYWORDS = {
    "python_skill": ["python", "pandas", "numpy", "scikit-learn", "sklearn", "matplotlib", "seaborn", "flask", "django", "fastapi", "tensorflow", "keras", "pytorch", "jupyter", "anaconda"],
    "ml_skill": ["machine learning", "deep learning", "neural network", "natural language processing", "nlp", "computer vision", "random forest", "decision tree", "knn", "svm", "logistic regression", "xgboost", "regression", "classification", "clustering", "recommendation system", "reinforcement learning"],
    "sql_skill": ["sql", "mysql", "postgresql", "sqlite", "oracle", "mongodb", "database", "nosql", "firebase", "redis", "queries", "stored procedure", "joins", "normalization"],
    "web_skill": ["html", "css", "javascript", "react", "angular", "vue", "node.js", "nodejs", "express", "rest api", "restful", "bootstrap", "tailwind", "typescript", "next.js", "php"],
}

INTEREST_KEYWORDS = {
    0: ["machine learning", "ai", "artificial intelligence", "data science", "deep learning", "nlp", "computer vision", "ml"],
    1: ["web development", "frontend", "backend", "full stack", "fullstack", "web design", "ui/ux", "react", "angular", "html", "css"],
    2: ["data analysis", "data analytics", "business intelligence", "bi", "tableau", "power bi", "excel", "visualization", "dashboard", "reporting", "data engineer"],
}

def extract_text_from_pdf(pdf_file) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("PyPDF2 not installed. Run: pip install PyPDF2")
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_name(text: str) -> str:
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    for line in lines[:3]:
        if 2 <= len(line.split()) <= 5 and re.match(r"^[A-Za-z\s\.\-]+$", line):
            return line.strip()
    return "Not Found"

def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not Found"

def extract_phone(text: str) -> str:
    match = re.search(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)(\d{3}[\s\-]?\d{4})", text)
    return match.group(0).strip() if match else "Not Found"

def extract_cgpa(text: str) -> float:
    patterns = [r"cgpa[:\s]*([0-9]\.[0-9]{1,2})", r"gpa[:\s]*([0-9]\.[0-9]{1,2})", r"([0-9]\.[0-9]{1,2})\s*/\s*10", r"([0-9]\.[0-9]{1,2})\s*/\s*4"]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = float(match.group(1))
            if val <= 4.0:
                val = round(val * 2.5, 2)
            return val
    return None

def extract_education(text: str) -> list:
    education = []
    degrees = ["b.tech", "btech", "b.e", "be ", "m.tech", "mtech", "mca", "bca", "b.sc", "m.sc", "bachelor", "master", "phd", "diploma"]
    lines = text.lower().splitlines()
    for i, line in enumerate(lines):
        if any(deg in line for deg in degrees):
            edu_line = text.splitlines()[i].strip()
            if edu_line and len(edu_line) > 5:
                education.append(edu_line)
    return education[:3]

def extract_experience(text: str) -> list:
    experience = []
    keywords = ["intern", "internship", "experience", "worked at", "engineer", "developer", "analyst", "trainee"]
    lines = text.splitlines()
    for line in lines:
        if any(kw in line.lower() for kw in keywords):
            clean = line.strip()
            if clean and len(clean) > 10:
                experience.append(clean)
    return experience[:5]

def extract_projects(text: str) -> list:
    projects = []
    lines = text.splitlines()
    in_projects = False
    for line in lines:
        line_lower = line.lower().strip()
        if "project" in line_lower and len(line_lower) < 20:
            in_projects = True
            continue
        if in_projects:
            if any(sec in line_lower for sec in ["education", "experience", "skill", "certification", "achievement", "award"]):
                break
            clean = line.strip()
            if clean and len(clean) > 10:
                projects.append(clean)
        if len(projects) >= 5:
            break
    return projects

def extract_certifications(text: str) -> list:
    certs = []
    keywords = ["certified", "certification", "certificate", "course", "coursera", "udemy", "nptel", "aws", "google", "microsoft"]
    for line in text.splitlines():
        if any(kw in line.lower() for kw in keywords):
            clean = line.strip()
            if clean and len(clean) > 10:
                certs.append(clean)
    return certs[:5]

def extract_skills(text: str) -> dict:
    text_lower = text.lower()
    result = {}
    detected = []
    for skill_key, keywords in SKILL_KEYWORDS.items():
        found = any(kw in text_lower for kw in keywords)
        result[skill_key] = 1 if found else 0
        if found:
            matched = [kw.title() for kw in keywords if kw in text_lower]
            detected.extend(matched[:2])
    interest_scores = {area: 0 for area in INTEREST_KEYWORDS}
    for area, keywords in INTEREST_KEYWORDS.items():
        interest_scores[area] = sum(1 for kw in keywords if kw in text_lower)
    result["interest"] = max(interest_scores, key=interest_scores.get)
    result["detected_skills"] = list(dict.fromkeys(detected))
    return result

def parse_resume(pdf_file) -> dict:
    text = extract_text_from_pdf(pdf_file)
    skills = extract_skills(text)
    cgpa = extract_cgpa(text)
    profile = {
        "name":           extract_name(text),
        "email":          extract_email(text),
        "phone":          extract_phone(text),
        "cgpa_detected":  cgpa,
        "education":      extract_education(text),
        "experience":     extract_experience(text),
        "projects":       extract_projects(text),
        "certifications": extract_certifications(text),
        **skills,
        "raw_text_length": len(text),
    }
    return profile

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resume_parser.py <resume.pdf>")
    else:
        profile = parse_resume(sys.argv[1])
        interest_map = {0: "ML / AI", 1: "Web Development", 2: "Data Analysis"}
        print("\n📄 FULL RESUME PROFILE")
        for k, v in profile.items():
            if k != "raw_text_length":
                print(f"  {k:20s}: {v}")