from __future__ import annotations
import tempfile
from pathlib import Path

import streamlit as st

from ml.skills import extract_skills
from ml.scoring import ats_skill_overlap
from ml.parsing import parse_resume, parse_job_description
from ml.skills import build_skill_taxonomy, keyword_gaps
from ml.scoring import compute_match, score_to_percent
from ml.explain import build_summary, recommendation_text

st.set_page_config(
    page_title="MatchrAI",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
# MatchrAI
### AI Resume Analyzer & ATS Job Matcher

Upload your resume, paste a job description, and receive an AI-powered compatibility report.""")

SKILLS_PATH = Path("data/skills_taxonomy.txt")
taxonomy = build_skill_taxonomy(SKILLS_PATH)

with st.sidebar:

    st.title("⚙ Settings")

    fresher_mode = st.toggle(
        "🎓 Fresher Mode",
        value=True
    )

    show_raw = st.toggle(
        "📄 Show Resume Text",
        value=False
    )

    st.divider()

    st.info(
        """
        **Tips**

        ✔ Upload PDF or DOCX

        ✔ Paste complete Job Description

        ✔ Larger taxonomy = Better ATS
        """)
   
    st.markdown("---")
    st.write("Tip: Expand `data/skills_taxonomy.txt` to improve skill extraction.")

left, right = st.columns([1, 1])

with left:

    st.subheader("📄 Upload Resume")

    resume_file = st.file_uploader(
        "",
        type=["pdf", "docx"]
    )
with right:

    st.subheader("💼 Job Description")

    jd_text = st.text_area(
        "",
        height=350,
        placeholder="Paste the job description here..."
    )

st.divider()

run = st.button(
    "🚀 Analyze Resume",
    use_container_width=True,
    type="primary"
)

if run:
    if resume_file is None:
        st.error("Please upload a resume (PDF or DOCX).")
        st.stop()

    if not jd_text.strip():
        st.error("Please paste the job description.")
        st.stop()

    with st.spinner("Analyzing Resume..."):
        progress = st.progress(0)
        suffix = "." + resume_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(resume_file.getbuffer())
            tmp_path = tmp.name

        progress.progress(20)
        parsed = parse_resume(tmp_path)

        progress.progress(40)
        jd_clean = parse_job_description(jd_text)

        # phase 3
        # -------------------------------
        # Extract skills FIRST
        # -------------------------------
        resume_sk = set(extract_skills(parsed.raw_text, taxonomy))
        jd_sk = set(extract_skills(jd_clean, taxonomy))

        # Debug (optional)
        #st.write("DEBUG jd_skills_count:", len(jd_sk))
        #st.write("DEBUG resume_skills_count:", len(resume_sk))
        #st.write("DEBUG overlap_count:", len(jd_sk & resume_sk))

        # -------------------------------
        # Compute ATS based on SKILLS
        # -------------------------------
        ats2 = ats_skill_overlap(resume_sk, jd_sk)

        #st.write("DEBUG ATS Skill Overlap:", ats2)

        # -------------------------------
        # Compute FINAL SCORE using override
        # -------------------------------
   
        progress.progress(70)

        scores = compute_match(
            resume_text=parsed.raw_text,
            jd_text=jd_clean,
            years_exp=parsed.years_experience_estimate,
            fresher_mode=fresher_mode,
            ats_override=ats2
        )
    
        # Debug statements 
        #st.write("DEBUG semantic_score:", scores.semantic_score)
        #st.write("DEBUG ats_score:", scores.ats_score)
        #st.write("DEBUG final_score:", scores.final_score)
        #st.write("DEBUG method:", scores.method)
    
        # Skill gaps
        gaps = keyword_gaps(parsed.raw_text, jd_clean, taxonomy)
        summary = build_summary(scores, gaps, parsed.cgpa)
        progress.progress(90)
        # Added for phase 3 to stop ATS from pulling score down

        ats2 = ats_skill_overlap(resume_sk, jd_sk)
   
        progress.progress(100)
        progress.empty()

    # Output sections outside the spinner but inside the button trigger
    st.markdown("---")
    topA, topB, topC = st.columns(3)
    topA.metric("Final Match Score", summary["final"])
    topB.metric("Semantic Similarity", summary["semantic"])
    topC.metric("ATS Skill Overlap", f"{int(round(ats2*100))}%")
  
    st.caption(f"Weighting used: semantic={scores.breakdown['weights']['semantic']:.2f}, ats={scores.breakdown['weights']['ats']:.2f} | "
               f"Estimated experience: {scores.breakdown['years_experience_estimate']} years | CGPA: {summary['cgpa']}")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Matched Skills (from Job Description)")
        if gaps["matched"]:
            st.success(", ".join(gaps["matched"]))
        else:
            st.info("No skills matched (based on current taxonomy). Add more skills to taxonomy or improve extraction.")

    with c2:
        st.subheader("Missing Skills (Skill Gap)")
        if gaps["missing"]:
            st.warning(", ".join(gaps["missing"]))
        else:
            st.success("No major missing skills detected!")

    st.subheader("Recommendations")
    st.write(recommendation_text(gaps["missing"]))

    st.subheader("Section Preview (Resume)")
    cols = st.columns(4)
    sec_names = ["education", "projects", "experience", "skills"]
    for i, sec in enumerate(sec_names):
        with cols[i]:
            st.markdown(f"**{sec.title()}**")
            st.write(parsed.sections.get(sec, "Not detected"))

    if show_raw:
        with st.expander("Show raw extracted resume text"):
            st.text(parsed.raw_text[:20000])

st.divider()

st.caption(
    "MatchrAI • Context-Aware SLM • Phase 3"
)
