from __future__ import annotations


RESUME_SIGNALS = {
    "education",
    "experience",
    "work experience",
    "employment",
    "skills",
    "technical skills",
    "projects",
    "certifications",
    "professional summary",
    "objective",
    "internship",
    "university",
    "college",
    "degree",
}

JD_SIGNALS = {
    "job title",
    "job overview",
    "responsibilities",
    "requirements",
    "qualifications",
    "required skills",
    "preferred skills",
    "preferred qualifications",
    "minimum qualifications",
    "education",
    "experience level",
    "about the role",
    "what you will do",
    "years of experience",
}

GENERAL_QUESTION_SIGNALS = {
    "what is the weather",
    "weather today",
    "who is the president",
    "latest election",
    "sports score",
    "who won the game",
    "stock price today",
    "tell me a recipe",
    "medical diagnosis",
}


def count_signals(text: str, signals: set[str]) -> int:
    normalized = " ".join(text.lower().split())
    return sum(1 for signal in signals if signal in normalized)


def looks_like_general_question(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(signal in normalized for signal in GENERAL_QUESTION_SIGNALS)


def guardrail_check(
    resume_text: str,
    jd_text: str
) -> tuple[bool, str]:

    resume_text = resume_text.strip()
    jd_text = jd_text.strip()

    if not resume_text:
        return False, "No readable resume text was extracted."

    if not jd_text:
        return False, "The job description is empty."

    if len(resume_text.split()) < 30:
        return False, (
            "The uploaded document contains too little readable text "
            "to be evaluated as a resume."
        )

    if len(jd_text.split()) < 20:
        return False, (
            "The job description is too short. Please paste the complete posting."
        )

    # Check only the JD input for a direct unrelated question.
    # Do not blacklist industry words in the resume.
    if looks_like_general_question(jd_text):
        return False, (
            "The entered text appears to be a general question rather than "
            "a job description."
        )

    resume_signal_count = count_signals(resume_text, RESUME_SIGNALS)
    jd_signal_count = count_signals(jd_text, JD_SIGNALS)

    if resume_signal_count < 1:
        return False, (
            "The uploaded document does not appear to contain common resume "
            "sections such as education, experience, skills, or projects."
        )

    if jd_signal_count < 1:
        return False, (
            "The entered text does not appear to contain job-description "
            "elements such as responsibilities, required skills, education, "
            "qualifications, or job overview."
        )

    return True, "Input accepted."