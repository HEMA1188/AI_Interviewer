import re # Added for potential text cleaning in future, or for regex if needed

class Prompts:
    """
    Collection of prompts used by the LLM for various interview-related tasks.
    These prompts are designed to be adaptable based on the candidate's Resume
    and the implied or explicit role/industry.
    """

    # --- 1. Interview Greeting ---
    greet_warmup_prompt = """
    Compose a welcoming greeting for the candidate at the start of the interview.

    Candidate's Name: {candidate_name}
    Candidate's Resume: {resume}
    Candidate's Actual Experience (e.g., Fresher, 1 year, 5 years): {candidate_actual_experience}
    Chosen Interview Difficulty Level: {chosen_interview_level}

    The greeting should be warm, professional, and set a positive tone for the interview.
    Mention the candidate's name and briefly acknowledge their background as presented in their **resume** and their **actual experience level**.
    Tailor the tone and content to acknowledge the *general professional nature* of the interview based on their resume, rather than assuming a specific technical field.
    Do not ask any questions here. Keep it to 2-3 sentences and do not end with a question mark.
    """

    # --- 1a. Skill Identification (Source: Resume) ---
    # This prompt is crucial for making the evaluation dynamic
    IDENTIFY_SKILL_PROMPT = """
    Given the following Candidate's Resume and an interview question:

    ---
    Candidate's Resume:
    {resume}
    ---

    Interview Question:
    {question}
    ---

    Based on the **candidate's resume** and the content of the **interview question**, identify the primary **professional skill, competency, or area of expertise** that this question is designed to assess.
    This could be a hard skill (e.g., 'Python', 'Financial Analysis', 'Digital Marketing', 'HR Policies') or a soft skill (e.g., 'Problem Solving', 'Communication Skills', 'Leadership', 'Client Management').
    Respond with only the skill name. Do not include any other text or explanation.
    If the question is very general, respond with 'General Professional Competency'.
    """

    # --- 1b. Skill Extraction from Resume (Refined for any field) ---
    EXTRACT_SKILLS_PROMPT = """
    Given the following candidate's resume, identify the **most prominent and relevant professional skills and competencies** demonstrated by the candidate that would be critical for interview assessment. These can be technical, domain-specific (e.g., marketing, finance, HR), or soft skills.

    Resume:
    {resume}

    List up to 10 key skills in a JSON array format like:
    {{"skills": ["Skill 1", "Skill 2", "Skill 3", ..., "Skill N"]}}
    Focus on skills that are strongly evidenced by their experience and responsibilities. Ensure the skills are relevant to professional roles based on the resume content.
    """

    # --- 2. Resume-Based Question Generation (Updated for any field) ---
    resume_based_question_prompt = """
    Generate an interview question specifically tailored to the **candidate's resume**, their **actual experience level ({candidate_actual_experience})**, and the **chosen interview difficulty level ({chosen_interview_level})**. The question should focus on one or more of the **{key_skills}** identified from their resume.

    Candidate's Resume:
    {resume}
    Candidate's Actual Experience: {candidate_actual_experience}
    Previous Questions Asked: {asked_questions}
    Chosen Interview Difficulty Level: {chosen_interview_level}
    Key Skills to Focus On: {key_skills}

    **Instructions for Question Complexity based on Chosen Interview Difficulty Level:**
    - If Chosen Interview Difficulty Level is 'Basic': Generate a simple, one-line question directly testing a skill from the resume, suitable for a candidate with {candidate_actual_experience} experience.
    - If Chosen Interview Difficulty Level is 'Intermediate': Generate a question that is 1 to 2 lines long, requiring practical application or explanation of a skill from the resume, suitable for a candidate with {candidate_actual_experience} experience.
    - If Chosen Interview Difficulty Level is 'Advanced': Generate a scenario-based question that encourages in-depth problem-solving, critical thinking, or strategic thinking related to skills from the resume, suitable for a candidate with {candidate_actual_experience} experience.

    The question should be highly relevant to the **candidate's listed experience and skills on their resume** AND directly related to one or more of the **Key Skills to Focus On**. It should be open-ended and encourage the candidate to provide a detailed response demonstrating their capabilities. Do not ask more than one question.
    """

    # --- 2b. Follow up Question based on previous answer (Updated for any field) ---
    follow_up_prompt = """
    Generate a follow-up question to the candidate's previous answer, suitable for a candidate with **{candidate_actual_experience}** experience at a **{chosen_interview_level}** difficulty. Ensure it connects to their resume and the **{key_skills}** identified.

    Previous Question: {previous_question}
    Candidate's Answer: {candidate_answer}
    Candidate's Actual Experience: {candidate_actual_experience}
    Chosen Interview Difficulty Level: {chosen_interview_level}
    Key Skills to Focus On: {key_skills}

    **Instructions for Follow-up Question Complexity based on Chosen Interview Difficulty Level:**
    - If Chosen Interview Difficulty Level is 'Basic': Generate a simple, one-line follow-up question for clarification or slight expansion of a previous point, appropriate for a {candidate_actual_experience} candidate.
    - If Chosen Interview Difficulty Level is 'Intermediate': Generate a follow-up question that is 1 to 2 lines long, delving deeper into a specific aspect of the answer, possibly involving a choice or approach, appropriate for a {candidate_actual_experience} candidate.
    - If Chosen Interview Difficulty Level is 'Advanced': Generate a challenging, scenario-based follow-up question that pushes for deeper analysis, alternative solutions, or broader implications related to their answer, appropriate for a {candidate_actual_experience} candidate.

    The follow-up question should seek to clarify, expand upon, or delve deeper into a specific aspect of the candidate's response, ideally touching upon one of the **Key Skills to Focus On**. It should aim to gain a more thorough understanding of their skills, experience, or thought process. Do not ask more than one question.
    """

    # --- 2c. Dynamic Question based on Resume and last answer (Updated for any field) ---
    resume_interview_prompt = """
    Generate a dynamic interview question based on the **candidate's resume**, the candidate's last answer, the questions that have already been asked, the candidate's **actual experience level ({candidate_actual_experience})**, the **chosen interview difficulty level ({chosen_interview_level})**, and the **{key_skills}** identified from their resume.

    Candidate's Resume:
    {resume}
    Candidate's Last Answer: {last_answer}
    Previous Questions Asked: {asked_questions}
    Number of Questions Asked: {question_count}
    Candidate's Actual Experience: {candidate_actual_experience}
    Chosen Interview Difficulty Level: {chosen_interview_level}
    Key Skills to Focus On: {key_skills}

    **Instructions for Question Complexity based on Chosen Interview Difficulty Level:**
    - If Chosen Interview Difficulty Level is 'Basic': Generate a simple, one-line question, appropriate for a {candidate_actual_experience} candidate.
    - If Chosen Interview Difficulty Level is 'Intermediate': Generate a question that is 1 to 2 lines long, based on the resume and previous answer, appropriate for a {candidate_actual_experience} candidate.
    - If Chosen Interview Difficulty Level is 'Advanced': Generate a scenario-based question that encourages in-depth problem-solving, critical thinking, or strategic thinking, building upon the interview flow, appropriate for a {candidate_actual_experience} candidate.

    The question should be highly relevant to the **candidate's background as described in their resume** and the flow of the interview, and should not be a repeat of a previous question. It MUST relate to one or more of the **Key Skills to Focus On**. Consider the candidate's last answer to create a natural conversation flow.
    """

    # --- 3. Response Handling (Silence, Irrelevant, No-Response) ---
    RESPONSE_CHECK_PROMPT = """
    Evaluate the following candidate answer to determine its relevance, silence, or if it is off-topic.

    Question: {question}

    Candidate's Answer: {candidate_answer}

    Output Format (JSON):
    {{
        "silence": "Yes" or "No",
        "irrelevant": "Yes" or "No",
        "notes": "Short reason if irrelevant or silent"
    }}
    """

    # --- 5. Closing & Candidate Questions ---
    closing_prompt = """
    The interview is now concluding. Please deliver a brief, professional closing statement, then ask the candidate:
    "Do you have any questions for us?"
    """

    # --- Question Generation (General, often initial or skill-specific) ---
    # This prompt needs significant generalization
    GENERATE_QUESTION_PROMPT = """
    You are an expert professional interviewer. Your goal is to generate a single, relevant interview question based on the candidate's resume, their **actual experience ({candidate_actual_experience})**, and the **chosen interview difficulty level ({chosen_interview_level})**.

    Candidate's Resume:
    {resume}
    Candidate's Actual Experience: {candidate_actual_experience}
    Candidate's Key Skills: {key_skills}
    Chosen Interview Difficulty Level: {chosen_interview_level}

    Guidelines:
    - Generate ONE question only.
    - The question should be challenging but appropriate for the '{chosen_interview_level}' difficulty level, given the candidate's '{candidate_actual_experience}' background.
    - Focus on skills explicitly mentioned in the 'Key Skills' list or clearly implied by the 'Resume'.
    - If a 'skill_to_assess' is provided, prioritize generating a question related to that specific skill.
    - The question should encourage detailed, practical answers, not just theoretical definitions.
    - Avoid asking questions that can be answered with a simple 'yes' or 'no'.
    - Ensure the question is clear, concise, and unambiguous.
    - **Crucially, the question should be relevant to the candidate's actual professional background as described in their resume, whether it's marketing, sales, finance, HR, teaching, or any other field, not just technical roles.**

    {skill_to_assess_instruction}

    Question:
    """

    # --- Candidate Answer Scoring Criteria (Updated for any field) ---
    SCORING_CRITERIA_PROMPT = """
    You are an expert interviewer evaluating a candidate's answer.

    Candidate's Resume: {resume}
    Candidate's Actual Experience: {candidate_actual_experience}
    Candidate's Key Skills: {key_skills}

    Chosen Interview Difficulty Level for this evaluation: {chosen_interview_level}

    Question: {question}
    Candidate's Answer: {candidate_answer}

    Based on the candidate's actual experience ({candidate_actual_experience}) and their resume, and considering the '{chosen_interview_level}' interview difficulty level:
    Evaluate the answer for:
    1.  **Relevance:** How well does the answer address the question, considering the candidate's professional background and the context of the interview?
    2.  **Depth:** Does the answer demonstrate appropriate professional depth for a candidate with '{candidate_actual_experience}' background, meeting the expectations of an '{chosen_interview_level}' interview?
    3.  **Accuracy:** Is the information presented factually correct and consistent with their resume?
    4.  **Completeness:** Is the answer comprehensive, covering necessary details without being overly verbose, for someone with their background and the given interview level?
    5.  **Communication:** How clearly, concisely, and effectively was the answer articulated? (e.g., structured thought, clarity of explanation, conciseness)

    Provide a score out of 5 for each criterion and an overall score out of 10. Also provide a detailed rationale.
    Output in JSON format:
    {{
        "relevance_score": int,
        "depth_score": int,
        "accuracy_score": int,
        "completeness_score": int,
        "communication_score": int,
        "overall_score": int,
        "rationale": "string"
    }}
    """

    # --- Final Interview Report Generation (Updated for any field) ---
    FINAL_REPORT_PROMPT = """
    You are an expert HR professional and interviewer. Generate a comprehensive final interview report based on the provided interview history, candidate's resume, their actual experience, and the chosen interview difficulty level.

    Candidate's Resume: {resume}
    Candidate's Actual Experience: {candidate_actual_experience}
    Chosen Interview Difficulty Level: {chosen_interview_level}
    Candidate's Key Skills: {key_skills}

    Interview History (Question and Answer pairs):
    {history}

    Structure the report as a professional dashboard, covering:
    1.  **Candidate Summary:** Briefly summarize the candidate's background based on their resume and actual experience ({candidate_actual_experience}), focusing on their professional trajectory and key achievements.
    2.  **Overall Assessment:** General strengths and weaknesses, considering their '{candidate_actual_experience}' and how they performed at the '{chosen_interview_level}' interview, in the context of their professional domain.
    3.  **Skill Assessment:** Evaluation of specific professional skills and competencies demonstrated during the interview, cross-referenced with their key skills identified from the resume.
    4.  **Performance per Question:** Briefly summarize performance on each question.
    5.  **Strengths:** Highlight notable strengths.
    6.  **Areas for Improvement:** Identify areas where the candidate could improve.
    7.  **Overall Recommendation:** (e.g., "Strong Hire," "Consider," "No Hire")
    8.  "recommendation_rationale": "string"
    9.  **Additional Comments:** Any additional insights or observations from the interview process relevant to their professional fit.

    Ensure the report is professional, objective, and aligns with the provided information, respecting the candidate's professional background regardless of its domain.
    """

    # --- Candidate Improvement Suggestions (Updated for any field) ---
    IMPROVEMENT_PROMPT = """
    You are an expert interview coach. Provide specific, actionable, and constructive ways a candidate could improve their interview answer to the given question, considering their actual experience and the chosen interview difficulty level.

    Candidate's Resume: {resume}
    Candidate's Actual Experience: {candidate_actual_experience}
    Chosen Interview Difficulty Level: {chosen_interview_level}

    Question: {question}
    Candidate's Answer: {candidate_answer}

    Provide only the suggestions, no conversational filler. Focus on clarity, completeness, relevance, and depth, tailored to someone with a '{candidate_actual_experience}' background aiming for a '{chosen_interview_level}' level performance within their professional domain.
    """

prompts = Prompts()