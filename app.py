from __future__ import annotations
import tempfile
from pathlib import Path

import streamlit as st
st.set_page_config(page_title="MatchrAI – Resume Analyzer & Job Matcher", layout="wide")

st.title("MatchrAI – Resume Analyzer & Job Match System (Capstone Project)")
st.caption("Upload a resume + paste a job description → get match score, ATS score, and skill gaps.")

from ml.guardrails import guardrail_check
from ml.skills import extract_skills
from ml.scoring import ats_skill_overlap
from ml.parsing import parse_resume, parse_job_description
from ml.skills import build_skill_taxonomy, keyword_gaps
from ml.scoring import compute_match, score_to_percent
from ml.explain import build_summary, recommendation_text
from ml.vector_store import ResumeVectorStore

@st.cache_resource
def get_vector_store(model_type: str) -> ResumeVectorStore:
    return ResumeVectorStore(model_type=model_type)



SKILLS_PATH = Path("data/skills_taxonomy.txt")
taxonomy = build_skill_taxonomy(SKILLS_PATH)


with st.sidebar:
    st.header("Settings")
    fresher_mode = st.toggle("Fresher / Student Mode", value=True, help="Weights semantic match higher for fresher resumes.")
    show_raw = st.toggle("Show extracted text", value=False)
    st.markdown("---")
    st.write("Tip: Expand `data/skills_taxonomy.txt` to improve skill extraction.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1) Upload Resume (PDF/DOCX)")
    resume_file = st.file_uploader("Resume file", type=["pdf", "docx"])
    st.subheader("2) Job Title")
    job_title = st.text_input(
    "Job title",
    placeholder="Example: Machine Learning Engineer"
    )

with col2:
    st.subheader("3) Job Description")
    jd_text = st.text_area("Paste job description here", height=220, placeholder="Paste the job description text...")

run = st.button("Analyze Match", type="primary", use_container_width=True)

if run:
    if resume_file is None:
        st.error("Please upload a resume (PDF or DOCX).")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste the job description text.")
        st.stop()

    # Save uploaded resume to temp file
    suffix = "." + resume_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(resume_file.getbuffer())
        tmp_path = tmp.name

    parsed = parse_resume(tmp_path)
    jd_clean = parse_job_description(jd_text)

    # Guardrails code
    # allowed, message = guardrail_check(parsed.raw_text, jd_clean)

    #if not allowed:
    #    st.error(message)
    #    st.stop()
    
    allowed, message = guardrail_check(parsed.raw_text, jd_clean)

    if not allowed:
        st.error("Request rejected by domain guardrails.")
        st.info(message)

        st.markdown("""
        **Supported inputs include:**
        - Resume in PDF or DOCX format
        - A complete job description
        - Resume-to-job matching
        - ATS skill-overlap analysis
        - Skill-gap analysis

        **Unsupported inputs include:**
        - General questions unrelated to employment
        - Weather questions
        - Political questions
        - Sports scores
        - Medical advice
        """)

        if show_raw:
            with st.expander("Debug extracted input"):
                st.write("Resume word count:", len(parsed.raw_text.split()))
                st.write("Job-description word count:", len(jd_clean.split()))
                st.text(parsed.raw_text[:2000])

        st.stop()

    
    # -------------------------------
    # Extract skills FIRST
    # -------------------------------
    resume_sk = set(extract_skills(parsed.raw_text, taxonomy))
    jd_sk = set(extract_skills(jd_clean, taxonomy))

    # -------------------------------
    # Store in Vector DB ( Chroma)
    # -------------------------------
    full_vector_store = None
    resume_id = None
    job_id = None

    try:
        full_vector_store = get_vector_store("full")

        resume_id = full_vector_store.add_resume(
            resume_text=parsed.raw_text,
            metadata={
                "filename": resume_file.name,
                "cgpa": parsed.cgpa or "Not detected",
                "years_experience": (
                    parsed.years_experience_estimate
                    if parsed.years_experience_estimate is not None
                    else -1
                )
            }
        )

        job_id = full_vector_store.add_job(
            jd_text=jd_clean,
            metadata={
                "source": "streamlit_input"
            }
        )

        st.caption(
            f"Stored in ChromaDB — Resume ID: {resume_id} | "
            f"Job ID: {job_id}"
        )

        st.success(
        "Resume and job-description embeddings were stored successfully."
        )

        with st.expander("Vector database details"):
            st.write("Resume ID:", resume_id)
            st.write("Job ID:", job_id)
            st.write("Embedding model:", "Full fine-tuned model")
            db_col1, db_col2 = st.columns(2)

        db_col1.metric(
            "Stored Resumes",
            full_vector_store.resume_count()
        )

        db_col2.metric(
            "Stored Job Descriptions",
            full_vector_store.job_count()
        )
    except Exception as exc:
        st.warning(
            "Resume matching completed, but vector database storage was unavailable."
        )

        if show_raw:
            st.exception(exc)

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
    
    # Commenting out for comparison of custom SLM and LoRA
     
    #scores = compute_match(
    #resume_text=parsed.raw_text,
    #jd_text=jd_clean,
    #years_exp=parsed.years_experience_estimate,
    #fresher_mode=fresher_mode,
    #ats_override=ats2   # important: pass the skill-based ATS score as an override to the main compute_match function
    #)
    
    scores_full = compute_match(
    resume_text=parsed.raw_text,
    jd_text=jd_clean,
    years_exp=parsed.years_experience_estimate,
    fresher_mode=fresher_mode,
    ats_override=ats2,
    model_type="full"
    )

    scores_lora = compute_match(
    resume_text=parsed.raw_text,
    jd_text=jd_clean,
    years_exp=parsed.years_experience_estimate,
    fresher_mode=fresher_mode,
    ats_override=ats2,
    model_type="lora"
    )
   
    # Skill gaps
    gaps = keyword_gaps(parsed.raw_text, jd_clean, taxonomy)

    # Debug statements 
    #st.write("DEBUG semantic_score:", scores.semantic_score)
    #st.write("DEBUG ats_score:", scores.ats_score)
    #st.write("DEBUG final_score:", scores.final_score)
    #st.write("DEBUG method:", scores.method)
    
    summary_full = build_summary(scores_full, gaps, parsed.cgpa)
    summary_lora = build_summary(scores_lora, gaps, parsed.cgpa)

    st.markdown("---")
    st.subheader("Model Comparison: Full Fine-Tuning vs LoRA / PEFT")

    col_full, col_lora = st.columns(2)

    with col_full:
        st.markdown("### Full Fine-Tuned Model")
        st.metric("Final Match Score", summary_full["final"])
        st.metric("Semantic Similarity", summary_full["semantic"])
        st.metric("ATS Skill Overlap", f"{int(round(ats2 * 100))}%")

    with col_lora:
        st.markdown("### LoRA / PEFT Model")
        st.metric("Final Match Score", summary_lora["final"])
        st.metric("Semantic Similarity", summary_lora["semantic"])
        st.metric("ATS Skill Overlap", f"{int(round(ats2 * 100))}%")

    delta = int(round(scores_lora.final_score * 100)) - int(round(scores_full.final_score * 100))
    st.info(f"LoRA vs Full Fine-Tuned Difference: {delta:+d} percentage points")

   

    # Commenting out for side by side comparison
    #summary = build_summary(scores, gaps, parsed.cgpa)

    # Added for phase 3 to stop ATS from pulling score down
    #resume_sk = set(extract_skills(parsed.raw_text, taxonomy))
    #jd_sk = set(extract_skills(jd_clean, taxonomy))

    #ats2 = ats_skill_overlap(resume_sk, jd_sk)
    
    # Commenting out for side by side comparison
    #st.markdown("---")
    #topA, topB, topC = st.columns(3)
    #topA.metric("Final Match Score", summary["final"])
    #topB.metric("Semantic Similarity", summary["semantic"])
    #topC.metric("ATS Skill Overlap", f"{int(round(ats2*100))}%")
    
    st.caption(
    f"Weighting used: semantic={scores_full.breakdown['weights']['semantic']:.2f}, "
    f"ats={scores_full.breakdown['weights']['ats']:.2f} | "
    f"Estimated experience: {parsed.years_experience_estimate or 'Not detected'} years | "
    f"CGPA: {summary_full['cgpa']}"
    )

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

    # Retrieve similar jobs from the vector database 
    if full_vector_store is not None:
        st.markdown("---")
        st.subheader("Top Similar Stored Job Descriptions")

        similar_jobs = full_vector_store.find_similar_jobs(
            resume_text=parsed.raw_text,
            top_k=6
        )

        job_ids = similar_jobs.get("ids", [[]])[0]
        job_metadata = similar_jobs.get("metadatas", [[]])[0]
        job_documents = similar_jobs.get("documents", [[]])[0]
        job_distances = similar_jobs.get("distances", [[]])[0]

        if not job_ids:
            st.info("No stored job descriptions are available yet.")
        else:
            displayed = 0

            for stored_job_id, metadata, document, distance in zip(
                job_ids,
                job_metadata,
                job_documents,
                job_distances
            ):
                # Do not show the same JD that was just submitted
                if stored_job_id == job_id:
                    continue

                displayed += 1

                with st.expander(
                    f"{displayed}. Stored Job — {stored_job_id}"
                ):
                    st.write("Metadata:", metadata)
                    st.write("Vector distance:", round(float(distance), 4))
                    st.text(document[:1500])

            if displayed == 0:
                st.info(
                    "No other stored job descriptions are available for comparison yet."
                )