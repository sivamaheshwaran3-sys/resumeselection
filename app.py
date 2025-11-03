import streamlit as st
import pandas as pd
import plotly.express as px
import random
import re
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# ------------------ Dark Theme ------------------
st.set_page_config(page_title="Resume Relevance Checker", layout="wide", page_icon="📄")

st.markdown("""
<style>
body, .stApp { background-color: #121212; color: white; }
div[data-testid="stSidebar"] { background-color: #1E1E1E; color: white; }
.stButton>button { background-color: #1565C0; color: white; }
</style>
""", unsafe_allow_html=True)

# ------------------ Session State ------------------
if "page" not in st.session_state: st.session_state.page = "upload"
if "jd_file" not in st.session_state: st.session_state.jd_file = None
if "resume_files" not in st.session_state: st.session_state.resume_files = None
if "results_df" not in st.session_state: st.session_state.results_df = None

# ------------------ Predefined Skills ------------------
PREDEFINED_SKILLS = [
    "Python","R","Pandas","NumPy","SQL","Spark","PySpark","TensorFlow","Keras","PyTorch",
    "Machine Learning","Deep Learning","NLP","Computer Vision","Data Visualization",
    "Tableau","Power BI","Excel","Git","Docker","Flask","FastAPI","API","Statistics",
    "Probability","Regression","Classification","Clustering","EDA","Data Cleaning","Databricks","Kafka"
]

SUGGESTION_TEMPLATES = [
    "Try working on {} projects to improve your skill.",
    "Consider learning {} to strengthen your profile.",
    "Enhance your expertise in {} for better chances.",
    "Gaining experience in {} would be valuable.",
    "Focus on developing skills in {} for better fit."
]

# ------------------ Text Extraction ------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text: text += page_text + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file):
    if file.name.endswith(".pdf"): return extract_text_from_pdf(file)
    elif file.name.endswith(".docx"): return extract_text_from_docx(file)
    return ""

# ------------------ Role & Skill Extraction ------------------
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

def extract_skills_from_jd(jd_text, skill_list=PREDEFINED_SKILLS):
    jd_text_lower = jd_text.lower()
    return [skill for skill in skill_list if skill.lower() in jd_text_lower]

def find_missing_skills(jd_skills, resume_text, max_suggestions=5):
    missing = [skill for skill in jd_skills if skill.lower() not in resume_text.lower()]
    return missing[:max_suggestions]

def generate_improvement_suggestion(missing_skills):
    if not missing_skills: return "All critical skills present!"
    suggestions = [random.choice(SUGGESTION_TEMPLATES).format(skill) for skill in missing_skills]
    wrapped_suggestions = []
    for s in suggestions:
        if len(s) > 50: wrapped_suggestions.append(s[:50] + "\n" + s[50:])
        else: wrapped_suggestions.append(s)
    return "\n".join(wrapped_suggestions)

# ------------------ Scoring ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

def compute_hard_score(jd_text, resume_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([jd_text, resume_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score * 100

def compute_soft_score(jd_text, resume_text):
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    score = util.cos_sim(jd_emb, resume_emb).item()
    return score * 100

def compute_hybrid_score(jd_text, resume_text, hard_weight=0.6, soft_weight=0.4):
    hard = compute_hard_score(jd_text, resume_text)
    soft = compute_soft_score(jd_text, resume_text)
    return round(hard*hard_weight + soft*soft_weight, 2)

def get_verdict(score):
    if score >= 70: return "High"
    elif score >= 40: return "Medium"
    else: return "Low"

# ------------------ Cached Resume Evaluation ------------------
@st.cache_data
def evaluate_resume(jd_text, resume_text):
    score = compute_hybrid_score(jd_text, resume_text)
    verdict = get_verdict(score)
    jd_skills = extract_skills_from_jd(jd_text)
    missing_skills = find_missing_skills(jd_skills, resume_text)
    suggestions = generate_improvement_suggestion(missing_skills)
    return {"hybrid_score": score, "verdict": verdict, "missing_skills": "\n".join(missing_skills) if missing_skills else "None", "suggestions": suggestions}

@st.cache_data
def process_files(jd_file, resume_files):
    jd_text = extract_text(jd_file)
    role_title = extract_role_title(jd_text)
    jd_skills = extract_skills_from_jd(jd_text)
    results = []
    for resume in resume_files:
        resume_text = extract_text(resume)
        evaluation = evaluate_resume(jd_text, resume_text)
        evaluation["Resume"] = resume.name
        results.append(evaluation)
    df = pd.DataFrame(results).sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    def assign_stars(idx): return "⭐⭐⭐" if idx==0 else "⭐⭐" if idx==1 else "⭐" if idx==2 else ""
    df["Top Star"] = [assign_stars(i) for i in range(len(df))]
    return jd_text, role_title, jd_skills, df

# ------------------ Upload Page ------------------
def page_upload():
    st.markdown("<h1 style='text-align:center;color:#BBDEFB'>📄 Automated Resume Relevance Checker</h1>", unsafe_allow_html=True)
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf","docx"])
    resume_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True)
    if jd_file and resume_files:
        st.session_state.jd_file = jd_file
        st.session_state.resume_files = resume_files
        st.session_state.page = "table"

# ------------------ Table Page ------------------
def page_table():
    jd_text, role_title, jd_skills, df = process_files(st.session_state.jd_file, st.session_state.resume_files)
    st.session_state.results_df = df

    # Job Info
    skill_badges = "".join([f"<span style='background-color:#1565C0;color:white;padding:6px 12px;margin:4px;border-radius:10px;display:inline-block;font-weight:bold;font-size:14px'>{s}</span>" for s in jd_skills])
    st.markdown(f"<div style='background-color:#2C2C2C;padding:20px;border-radius:15px;text-align:center'><h2>💼 {role_title}</h2><p><b>Key Skills:</b><br>{skill_badges}</p></div>", unsafe_allow_html=True)

    # Top 3 Cards
    st.subheader("🌟 Top 3 Candidates")
    top3 = df.head(3)
    cols = st.columns(3)
    for col, row in zip(cols, top3.to_dict("records")):
        col.markdown(f"<div style='background-color:#2C2C2C;padding:15px;border-radius:15px;'><h3>{row['Resume']} {row['Top Star']}</h3><p><b>Score:</b> {row['hybrid_score']}</p><p><b>Missing Skills:</b><br>{row['missing_skills']}</p><p><b>Suggestions:</b><br>{row['suggestions']}</p></div>", unsafe_allow_html=True)

    # Full Table
    st.subheader("📊 All Resumes")
    st.data_editor(df, hide_index=True, column_config={
        "Top Star": st.column_config.TextColumn(width="small"),
        "suggestions": st.column_config.TextColumn(width="large"),
        "missing_skills": st.column_config.TextColumn(width="medium")
    }, height=600)

    if st.button("📈 View Graph"): st.session_state.page = "graph"

# ------------------ Graph Page ------------------
def page_graph():
    df = st.session_state.results_df
    st.subheader("📈 Hybrid Score Chart")
    fig = px.bar(df.sort_values("hybrid_score"), x="hybrid_score", y="Resume", orientation="h", text="hybrid_score",
                 color="hybrid_score", color_continuous_scale="Blues", template="plotly_dark")
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='#121212', paper_bgcolor='#121212', font_color='white', margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    if st.button("⬅ Back to Table"): st.session_state.page = "table"
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 Download CSV", data=csv, file_name="resume_relevance_results.csv", mime="text/csv")

# ------------------ Page Router ------------------
if st.session_state.page == "upload": page_upload()
elif st.session_state.page == "table": page_table()
elif st.session_state.page == "graph": page_graph()
