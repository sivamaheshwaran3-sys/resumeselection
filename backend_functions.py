import re
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# ------------------ Model ------------------
_model = None

def load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

# ------------------ File Text Extraction ------------------
def extract_text_from_pdf(file_path_or_obj):
    text = ""
    with pdfplumber.open(file_path_or_obj) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_path_or_obj):
    doc = Document(file_path_or_obj)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file):
    """
    file: file object (PDF/DOCX)
    """
    if hasattr(file, "name") and file.name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif hasattr(file, "name") and file.name.endswith(".docx"):
        return extract_text_from_docx(file)
    else:
        return ""

# ------------------ Role Extraction ------------------
def extract_role_title(jd_text):
    lines = [line.strip("• ").strip() for line in jd_text.split("\n") if line.strip()]
    role_keywords = ["Intern", "Engineer", "Developer", "Analyst", "Manager", "Scientist", "Data", "Machine Learning"]
    ignore_keywords = ["Duration", "Bond", "Stipend", "Schedule", "Location", "Eligibility", "Qualification"]
    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in role_keywords) and \
           not any(ignore.lower() in line.lower() for ignore in ignore_keywords):
            return line
    for line in lines:
        if re.search(r"(data|analytics|manufacturing|engineer|scientist|developer|intern)", line, re.I):
            return line
    return "Role not detected"

# ------------------ Skill Extraction ------------------
PREDEFINED_SKILLS = [
    "Python","R","Pandas","NumPy","SQL","Spark","PySpark","TensorFlow","Keras","PyTorch",
    "Machine Learning","Deep Learning","NLP","Computer Vision","Data Visualization",
    "Tableau","Power BI","Excel","Git","Docker","Flask","FastAPI","API","Statistics",
    "Probability","Regression","Classification","Clustering","EDA","Data Cleaning","Databricks","Kafka"
]

def extract_skills_from_jd(jd_text, skill_list=PREDEFINED_SKILLS):
    jd_text_lower = jd_text.lower()
    return [skill for skill in skill_list if skill.lower() in jd_text_lower]

def find_missing_skills(jd_skills, resume_text, max_suggestions=5):
    missing = [skill for skill in jd_skills if skill.lower() not in resume_text.lower()]
    return missing[:max_suggestions]

SUGGESTION_TEMPLATES = [
    "Learn {}",
    "Practice {}",
    "Work on {}",
    "Improve {} skills",
    "Focus on {}"
]

def generate_improvement_suggestion(missing_skills):
    if not missing_skills:
        return "All critical skills present!"
    return "\n".join([random.choice(SUGGESTION_TEMPLATES).format(skill) for skill in missing_skills])

# ------------------ Scoring ------------------
def compute_hard_score(jd_text, resume_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([jd_text, resume_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score * 100

def compute_soft_score(jd_text, resume_text):
    model = load_model()
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    return util.cos_sim(jd_emb, resume_emb).item() * 100

def compute_hybrid_score(jd_text, resume_text, hard_weight=0.6, soft_weight=0.4):
    hard = compute_hard_score(jd_text, resume_text)
    soft = compute_soft_score(jd_text, resume_text)
    return round(hard*hard_weight + soft*soft_weight, 2)

def get_verdict(score):
    if score >= 70: return "High"
    elif score >= 40: return "Medium"
    else: return "Low"

# ------------------ Full Evaluation ------------------
def evaluate_resume(jd_text, resume_text):
    """
    Returns full evaluation dictionary for a resume
    """
    score_hard = compute_hard_score(jd_text, resume_text)
    score_soft = compute_soft_score(jd_text, resume_text)
    score_hybrid = compute_hybrid_score(jd_text, resume_text)

    jd_skills = extract_skills_from_jd(jd_text)
    missing_skills = find_missing_skills(jd_skills, resume_text)
    suggestions = generate_improvement_suggestion(missing_skills)
    verdict = get_verdict(score_hybrid)

    return {
        "hard_score": round(score_hard,2),
        "soft_score": round(score_soft,2),
        "hybrid_score": score_hybrid,
        "verdict": verdict,
        "missing_skills": "\n".join(missing_skills) if missing_skills else "None",
        "suggestions": suggestions
    }
