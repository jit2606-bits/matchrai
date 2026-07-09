import re

RESUME_JD_KEYWORDS = {
    "resume", "cv", "job description", "jd", "skills", "experience",
    "education", "projects", "certifications", "candidate", "role",
    "responsibilities", "requirements", "qualifications", "ats",
    "match score", "semantic similarity", "missing skills",
    "matched skills", "hiring", "recruiter", "job posting"
}

OUT_OF_SCOPE_KEYWORDS = {
    "weather", "temperature", "rain", "snow", "storm",
    "politics", "election", "president", "senate", "congress",
    "sports", "nba", "nfl", "mlb", "soccer", "cricket",
    "stock price", "crypto", "bitcoin",
    "medical advice", "diagnosis", "medicine",
    "recipe", "movie", "celebrity"
}

def is_resume_matching_task(resume_text: str, jd_text: str) -> bool:
    combined = f"{resume_text} {jd_text}".lower()

    resume_signals = [
        "education", "experience", "skills", "projects",
        "certifications", "technical skills", "work experience"
    ]

    jd_signals = [
        "requirements", "responsibilities", "qualifications",
        "job description", "role", "preferred qualifications",
        "minimum qualifications"
    ]

    has_resume_signal = any(s in combined for s in resume_signals)
    has_jd_signal = any(s in combined for s in jd_signals)

    return has_resume_signal and has_jd_signal


def detect_out_of_scope(text: str) -> bool:
    low = text.lower()
    return any(keyword in low for keyword in OUT_OF_SCOPE_KEYWORDS)


def guardrail_check(resume_text: str, jd_text: str):
    combined = f"{resume_text} {jd_text}"

    if detect_out_of_scope(combined):
        return False, (
            "This application is designed only for resume-to-job-description matching. "
            "It does not answer general questions about weather, politics, sports, finance, "
            "medical topics, or unrelated subjects. Please upload a resume and paste a job description."
        )

    if not is_resume_matching_task(resume_text, jd_text):
        return False, (
            "The provided input does not look like a valid resume and job description pair. "
            "Please upload a resume and paste a job description containing role requirements, "
            "responsibilities, qualifications, or skills."
        )

    return True, "Input accepted."