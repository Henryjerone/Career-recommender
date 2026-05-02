"""
app.py — AI Career Recommendation System
Upgraded: Smart predictions + Premium UI
"""

import streamlit as st
import pickle
import numpy as np
from resume_parser import parse_resume

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="CareerAI — Smart Career Recommender",
    page_icon="🎯",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Premium Dark UI
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C14;
    color: #E8EDF5;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 50%, #0D1B2A 100%);
    border: 1px solid #1E3A5F;
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(0,180,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(0,180,255,0.1);
    border: 1px solid rgba(0,180,255,0.3);
    color: #00B4FF;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.15;
    margin: 0.5rem 0;
}
.hero h1 span { color: #00B4FF; }
.hero p {
    color: #7A9BBE;
    font-size: 1.05rem;
    font-weight: 300;
    margin-top: 0.8rem;
    max-width: 500px;
}

/* Section cards */
.section-card {
    background: #0D1B2A;
    border: 1px solid #1A2E45;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.02em;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Profile card after parsing */
.profile-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.profile-item {
    background: #0A1628;
    border: 1px solid #1A2E45;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
.profile-item .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4A7098;
    margin-bottom: 0.2rem;
}
.profile-item .value {
    font-size: 0.95rem;
    color: #C8D8E8;
    font-weight: 500;
}

/* Skill tag */
.skill-tag {
    display: inline-block;
    background: rgba(0,180,255,0.08);
    border: 1px solid rgba(0,180,255,0.2);
    color: #5BC8F5;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    margin: 0.2rem;
}
.skill-tag.found { background: rgba(0,210,120,0.08); border-color: rgba(0,210,120,0.25); color: #2ECC71; }
.skill-tag.missing { background: rgba(255,80,80,0.06); border-color: rgba(255,80,80,0.15); color: #E05050; }

/* List items */
.info-list { list-style: none; padding: 0; margin: 0.5rem 0; }
.info-list li {
    padding: 0.4rem 0;
    color: #8AAFC8;
    font-size: 0.9rem;
    border-bottom: 1px solid #111E2E;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
}
.info-list li::before { content: '→'; color: #00B4FF; flex-shrink: 0; }

/* Recommendation output */
.rec-wrapper {
    background: linear-gradient(135deg, #081422 0%, #0D1B2A 100%);
    border: 1px solid #1E3A5F;
    border-radius: 20px;
    padding: 2.5rem;
    margin-top: 1.5rem;
}
.rec-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00B4FF;
    margin-bottom: 0.5rem;
}
.rec-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0;
}
.rec-sub {
    color: #5A8AAA;
    font-size: 0.9rem;
    margin-top: 0.3rem;
}
.confidence-bar-bg {
    background: #111E2E;
    border-radius: 6px;
    height: 6px;
    margin: 1rem 0 0.3rem;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #00B4FF, #0072FF);
}
.confidence-label {
    font-size: 0.78rem;
    color: #4A7098;
}

/* Score cards */
.score-card {
    background: #0A1628;
    border: 1px solid #1A2E45;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.score-card .num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #00B4FF;
}
.score-card .lbl {
    font-size: 0.75rem;
    color: #4A7098;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Insight boxes */
.insight-box {
    background: #0A1628;
    border-left: 3px solid #00B4FF;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #8AAFC8;
    line-height: 1.5;
}
.insight-box strong { color: #C8D8E8; }

/* Divider */
.div { border: none; border-top: 1px solid #1A2E45; margin: 1.5rem 0; }

/* Button override */
.stButton > button {
    background: linear-gradient(135deg, #0072FF, #00B4FF) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(0,114,255,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,114,255,0.45) !important;
}

/* Input labels */
label { color: #7A9BBE !important; font-size: 0.85rem !important; }
.stSlider > div { padding: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

bundle        = load_model()
model         = bundle["internship_model"]
proj_model    = bundle["project_model"]
scaler        = bundle["scaler"]
le_internship = bundle["le_internship"]
le_project    = bundle["le_project"]

# ─────────────────────────────────────────────
# SMARTER SCORING FUNCTION
# ─────────────────────────────────────────────

def compute_career_score(cgpa, python_skill, ml_skill, sql_skill, web_skill,
                          interest, experience, certifications, projects):
    score = 0
    breakdown = {}

    # CGPA score (max 30)
    cgpa_pts = round(min(cgpa / 10.0, 1.0) * 30, 1)
    breakdown["Academic (CGPA)"] = cgpa_pts
    score += cgpa_pts

    # Skills score (max 25)
    skills_pts = sum([python_skill, ml_skill, sql_skill, web_skill]) * 6.25
    breakdown["Technical Skills"] = round(skills_pts, 1)
    score += skills_pts

    # Experience score (max 20)
    exp_pts = min(len(experience), 3) * 6.66
    breakdown["Experience"] = round(exp_pts, 1)
    score += exp_pts

    # Certifications score (max 15)
    cert_pts = min(len(certifications), 3) * 5.0
    breakdown["Certifications"] = round(cert_pts, 1)
    score += cert_pts

    # Projects score (max 10)
    proj_pts = min(len(projects), 2) * 5.0
    breakdown["Projects"] = round(proj_pts, 1)
    score += proj_pts

    return round(min(score, 100), 1), breakdown


def get_career_insights(cgpa, python_skill, ml_skill, sql_skill, web_skill,
                         interest, experience, certifications, projects, internship_label):
    insights = []
    tips = []

    interest_map = {0: "ML/AI", 1: "Web Development", 2: "Data Analysis"}
    area = interest_map[interest]

    # Strength insights
    if cgpa >= 8.5:
        insights.append(f"🌟 <strong>Strong academic profile</strong> — your CGPA of {cgpa} stands out to recruiters.")
    if len(experience) >= 1:
        insights.append(f"💼 <strong>{len(experience)} experience entry(ies) detected</strong> — practical exposure boosts your profile significantly.")
    if len(certifications) >= 1:
        insights.append(f"🏅 <strong>{len(certifications)} certification(s) found</strong> — verified credentials validate your skills.")
    if len(projects) >= 2:
        insights.append(f"🛠️ <strong>{len(projects)} projects detected</strong> — a strong project portfolio demonstrates hands-on ability.")

    # Skill tips
    if not python_skill and interest in [0, 2]:
        tips.append("Learn Python — it is the #1 skill for ML and Data Analysis roles.")
    if not ml_skill and interest == 0:
        tips.append("Add ML frameworks like Scikit-learn or TensorFlow to strengthen your AI profile.")
    if not sql_skill:
        tips.append("SQL is expected in almost every data role — add it to your toolkit.")
    if not web_skill and interest == 1:
        tips.append("Add React or Node.js to your skill set for Web Development internships.")
    if len(certifications) == 0:
        tips.append("Complete a free certification (Coursera, NPTEL, or Google) to validate your skills.")
    if len(projects) < 2:
        tips.append(f"Build at least 2 projects related to {area} before applying.")
    if cgpa < 7.0:
        tips.append("Focus on improving your CGPA — many companies have a 7.0+ cutoff.")

    return insights, tips


# ─────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-tag">AI Powered · ML Based</div>
    <h1>Career<span>AI</span></h1>
    <h1 style="font-size:1.6rem; font-weight:600; color:#7A9BBE; margin-top:-0.3rem;">
        Smart Internship & Project Recommender
    </h1>
    <p>Upload your resume — our AI extracts your complete profile and recommends the best career path using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT: Two columns
# ─────────────────────────────────────────────

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:

    # ── RESUME UPLOAD ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📄 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("PDF format only", type=["pdf"], label_visibility="collapsed")

    auto = {}
    if uploaded_file:
        with st.spinner("Analysing resume..."):
            auto = parse_resume(uploaded_file)
        st.success("✅ Resume analysed!")

        # Personal info grid
        st.markdown('<div class="profile-grid">', unsafe_allow_html=True)
        fields = [
            ("Name", auto.get("name","—")),
            ("Email", auto.get("email","—")),
            ("Phone", auto.get("phone","—")),
            ("CGPA", auto.get("cgpa_detected","—")),
        ]
        html_items = ""
        for label, val in fields:
            html_items += f'<div class="profile-item"><div class="label">{label}</div><div class="value">{val}</div></div>'
        st.markdown(html_items + '</div>', unsafe_allow_html=True)

        # Skills tags
        skill_html = ""
        for key, label in [("python_skill","Python"), ("ml_skill","ML/AI"), ("sql_skill","SQL"), ("web_skill","Web Dev")]:
            cls = "found" if auto.get(key) else "missing"
            icon = "✓" if auto.get(key) else "✗"
            skill_html += f'<span class="skill-tag {cls}">{icon} {label}</span>'
        st.markdown(skill_html, unsafe_allow_html=True)

        # Education
        if auto.get("education"):
            st.markdown("<br><div style='font-size:0.78rem;color:#4A7098;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem'>Education</div>", unsafe_allow_html=True)
            items = "".join([f"<li>{e}</li>" for e in auto["education"]])
            st.markdown(f'<ul class="info-list">{items}</ul>', unsafe_allow_html=True)

        # Experience
        if auto.get("experience"):
            st.markdown("<div style='font-size:0.78rem;color:#4A7098;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;margin-top:0.8rem'>Experience</div>", unsafe_allow_html=True)
            items = "".join([f"<li>{e}</li>" for e in auto["experience"][:3]])
            st.markdown(f'<ul class="info-list">{items}</ul>', unsafe_allow_html=True)

        # Projects
        if auto.get("projects"):
            st.markdown("<div style='font-size:0.78rem;color:#4A7098;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;margin-top:0.8rem'>Projects</div>", unsafe_allow_html=True)
            items = "".join([f"<li>{p}</li>" for p in auto["projects"][:3]])
            st.markdown(f'<ul class="info-list">{items}</ul>', unsafe_allow_html=True)

        # Certifications
        if auto.get("certifications"):
            st.markdown("<div style='font-size:0.78rem;color:#4A7098;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;margin-top:0.8rem'>Certifications</div>", unsafe_allow_html=True)
            items = "".join([f"<li>{c}</li>" for c in auto["certifications"][:3]])
            st.markdown(f'<ul class="info-list">{items}</ul>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── MANUAL INPUT ──
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Your Details</div>', unsafe_allow_html=True)

    name = st.text_input("Full Name", value=(auto.get("name","") if auto.get("name","") not in ["Not Found",""] else ""), placeholder="e.g. Priya Sharma")
    cgpa = st.slider("CGPA", 0.0, 10.0, float(auto.get("cgpa_detected") or 7.5), step=0.1)

    st.markdown("<div style='font-size:0.82rem;color:#4A7098;margin:0.8rem 0 0.4rem'>Technical Skills</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        python_skill = st.checkbox("Python / DS", value=bool(auto.get("python_skill", 0)))
        ml_skill     = st.checkbox("Machine Learning", value=bool(auto.get("ml_skill", 0)))
    with c2:
        sql_skill    = st.checkbox("SQL / Database", value=bool(auto.get("sql_skill", 0)))
        web_skill    = st.checkbox("Web Dev", value=bool(auto.get("web_skill", 0)))

    interest = st.selectbox("Primary Interest",
        options=[0, 1, 2],
        format_func=lambda x: {0:"🤖 ML / AI", 1:"🌐 Web Development", 2:"📊 Data Analysis"}[x],
        index=int(auto.get("interest", 0))
    )

    st.markdown('</div>', unsafe_allow_html=True)
    recommend_btn = st.button("⚡ Generate My Career Recommendation", use_container_width=True)

# ─────────────────────────────────────────────
# RIGHT COLUMN — OUTPUT
# ─────────────────────────────────────────────

with right_col:

    if recommend_btn:
        if not name:
            st.warning("Please enter your name.")
        else:
            experience     = auto.get("experience", [])
            certifications = auto.get("certifications", [])
            projects       = auto.get("projects", [])

            # ML prediction
            features = np.array([[cgpa, int(python_skill), int(ml_skill),
                                   int(sql_skill), int(web_skill), interest]])
            features_scaled    = scaler.transform(features)
            internship_pred    = model.predict(features_scaled)[0]
            internship_prob    = model.predict_proba(features_scaled).max()
            project_pred       = proj_model.predict(features_scaled)[0]
            project_prob       = proj_model.predict_proba(features_scaled).max()
            internship_label   = le_internship.inverse_transform([internship_pred])[0]
            project_label      = le_project.inverse_transform([project_pred])[0]

            # Career score
            career_score, breakdown = compute_career_score(
                cgpa, python_skill, ml_skill, sql_skill, web_skill,
                interest, experience, certifications, projects
            )

            # Insights
            insights, tips = get_career_insights(
                cgpa, python_skill, ml_skill, sql_skill, web_skill,
                interest, experience, certifications, projects, internship_label
            )

            # ── Score Cards ──
            s1, s2, s3 = st.columns(3)
            score_color = "#2ECC71" if career_score >= 70 else "#F39C12" if career_score >= 50 else "#E74C3C"
            with s1:
                st.markdown(f'<div class="score-card"><div class="num" style="color:{score_color}">{career_score}</div><div class="lbl">Career Score</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="score-card"><div class="num">{cgpa}</div><div class="lbl">CGPA</div></div>', unsafe_allow_html=True)
            with s3:
                skill_count = sum([int(python_skill), int(ml_skill), int(sql_skill), int(web_skill)])
                st.markdown(f'<div class="score-card"><div class="num">{skill_count}/4</div><div class="lbl">Skills</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Internship Recommendation ──
            fill_pct = int(internship_prob * 100)
            st.markdown(f"""
            <div class="rec-wrapper">
                <div class="rec-header">💼 Recommended Internship</div>
                <div class="rec-title">{internship_label}</div>
                <div class="rec-sub">Best fit based on your skills, CGPA & interest area</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{fill_pct}%"></div>
                </div>
                <div class="confidence-label">Model Confidence: {fill_pct}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Project Recommendation ──
            fill_pct2 = int(project_prob * 100)
            st.markdown(f"""
            <div class="rec-wrapper" style="border-color:#1A3A2A;">
                <div class="rec-header" style="color:#2ECC71;">🛠️ Recommended Project</div>
                <div class="rec-title">{project_label}</div>
                <div class="rec-sub">Ideal project to build for your target domain</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{fill_pct2}%; background: linear-gradient(90deg, #2ECC71, #27AE60);"></div>
                </div>
                <div class="confidence-label">Model Confidence: {fill_pct2}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Profile Breakdown ──
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📊 Score Breakdown</div>', unsafe_allow_html=True)
            for category, pts in breakdown.items():
                max_pts = {"Academic (CGPA)": 30, "Technical Skills": 25, "Experience": 20, "Certifications": 15, "Projects": 10}[category]
                pct = int((pts / max_pts) * 100)
                bar_color = "#00B4FF" if pct >= 70 else "#F39C12" if pct >= 40 else "#E74C3C"
                st.markdown(f"""
                <div style="margin-bottom:0.8rem;">
                    <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#7A9BBE;margin-bottom:0.3rem;">
                        <span>{category}</span><span>{pts} / {max_pts}</span>
                    </div>
                    <div style="background:#111E2E;border-radius:4px;height:5px;overflow:hidden;">
                        <div style="width:{pct}%;height:100%;background:{bar_color};border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Insights ──
            if insights:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">✨ Profile Strengths</div>', unsafe_allow_html=True)
                for i in insights:
                    st.markdown(f'<div class="insight-box">{i}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Tips ──
            if tips:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🚀 How to Improve</div>', unsafe_allow_html=True)
                for t in tips:
                    st.markdown(f'<div class="insight-box" style="border-left-color:#F39C12;">{t}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style="height:100%;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:4rem 2rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">🎯</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:#FFFFFF;margin-bottom:0.5rem;">
                Your Career Report
            </div>
            <div style="color:#4A7098;font-size:0.92rem;max-width:280px;line-height:1.6;">
                Upload your resume and click <strong style="color:#7A9BBE">Generate</strong> to see your personalized internship recommendation, project suggestion, career score, and improvement tips.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:2rem 0 0;color:#2A4A6A;font-size:0.78rem;letter-spacing:0.05em;">
    CareerAI — Major Project · Built with Streamlit & Scikit-learn · ML Powered
</div>
""", unsafe_allow_html=True)