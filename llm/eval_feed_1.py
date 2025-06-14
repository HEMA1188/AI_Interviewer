import asyncio
import json
import logging
import os
import sys
from typing import List, Optional, Dict
import uuid
import traceback # Import traceback for more detailed error logging
import tempfile # For atomic file writes


# === Setup Project Path (adjust if needed) ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === Imports ===
# Assuming llm_task is the correct path to your LLMHandler
from llm.llm_task_1 import LLMHandler
from resources.prompts_3 import prompts

# Import the client from config.py
from utils.config import client as openai_client, DEFAULT_MODEL


# === Logging Setup ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Import the PDF Extractor
try:
    from utils.pdf_extractor import extract_text_from_pdf
except ImportError:
    # Handle the case where pdf_extractor might not be available or path is wrong
    logger.error("Could not import 'extract_text_from_pdf' from 'utils.pdf_extractor'. "
                 "Ensure pdf_extractor.py exists in utils/ and its dependencies are installed (e.g., pypdf).")
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Placeholder if PDF extractor is not available."""
        logger.warning(f"PDF extraction not available. Cannot extract text from {pdf_path}.")
        return ""


# === Configuration ===
INTERVIEW_DATA_FILE = os.getenv("INTERVIEW_DATA_FILE", "Candidates_interview_database_1.json")


# --- File Handling Functions ---

async def load_interview_data(candidate_id: str) -> Dict:
    """
    Loads interview data for a specific candidate from the JSON file.
    Returns an empty dict if the file isn't found or if there's a JSON error.
    """
    try:
        # Ensure the file exists, initialize with empty JSON object if it doesn't
        if not os.path.exists(INTERVIEW_DATA_FILE):
            logger.info(f"Creating new interview data file: {INTERVIEW_DATA_FILE}")
            with open(INTERVIEW_DATA_FILE, 'w') as f:
                json.dump({}, f)

        with open(INTERVIEW_DATA_FILE, 'r') as f:
            file_content = f.read().strip()
            if not file_content: # Handle empty file case
                logger.warning(f"File {INTERVIEW_DATA_FILE} is empty. Initializing data structure.")
                all_data = {}
            else:
                all_data = json.loads(file_content)
                # Ensure the top-level is a dict
                if not isinstance(all_data, dict):
                    logger.error(f"Root of {INTERVIEW_DATA_FILE} is not a dictionary. Reinitializing.")
                    all_data = {}
            return all_data.get(candidate_id, {"questions": {}, "answers": {}, "evaluations": {}})
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {INTERVIEW_DATA_FILE}: {e}. Returning empty data and potentially corrupting if saved over.")
        return {"questions": {}, "answers": {}, "evaluations": {}}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading interview data: {e}\n{traceback.format_exc()}")
        return {"questions": {}, "answers": {}, "evaluations": {}}

async def save_interview_data(candidate_id: str, data: Dict):
    """
    Saves interview data for a specific candidate to the JSON file using an atomic write.
    Handles file creation and JSON errors.
    """
    all_data = {}
    try:
        # Load existing data if file exists and is not empty
        if os.path.exists(INTERVIEW_DATA_FILE):
            with open(INTERVIEW_DATA_FILE, 'r') as f:
                file_content = f.read().strip()
                if file_content:
                    all_data = json.loads(file_content)
                    if not isinstance(all_data, dict):
                        logger.error(f"Root of {INTERVIEW_DATA_FILE} is not a dictionary during save load. Overwriting.")
                        all_data = {}
                else:
                    logger.warning(f"File {INTERVIEW_DATA_FILE} is empty during save load. Initializing data structure.")
        else:
            logger.info(f"Creating new interview data file for saving: {INTERVIEW_DATA_FILE}")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {INTERVIEW_DATA_FILE} during save load: {e}. Initializing new data structure.")
        all_data = {} # Re-initialize if current file is corrupt
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading for save: {e}\n{traceback.format_exc()}")
        all_data = {}

    all_data[candidate_id] = data # Update the specific candidate's entry

    try:
        # Use a temporary file for atomic write
        # This prevents data loss or corruption if the application crashes during write
        fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(INTERVIEW_DATA_FILE))
        with os.fdopen(fd, 'w') as tmp_f:
            json.dump(all_data, tmp_f, indent=4)
        os.replace(temp_path, INTERVIEW_DATA_FILE) # Atomically replace the original file
        logger.info(f"Interview data for candidate '{candidate_id}' saved to {INTERVIEW_DATA_FILE}")
    except Exception as e:
        logger.error(f"Failed to write to {INTERVIEW_DATA_FILE}: {e}\n{traceback.format_exc()}")
        # Clean up temp file if something went wrong before replace
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- Interview Evaluation and Feedback Functions ---

async def evaluate_candidate_answer(handler: LLMHandler, candidate_id: str, question_id: str, question_text: str, answer: str, skill_being_assessed: Optional[str]) -> Dict:
    """
    Evaluates the candidate's answer and stores the result.
    The evaluation is purely based on the resume and experience level stored in the LLMHandler.
    """
    try:
        # LLMHandler.score_candidate_answer automatically uses its internal candidate_experience_level,
        # resume, and key skills.
        evaluation = await handler.score_candidate_answer(question_text, answer, skill_being_assessed)
        
        data = await load_interview_data(candidate_id)
        data["questions"][question_id] = question_text
        data["answers"][question_id] = answer
        data["evaluations"][question_id] = {
            "question": question_text,
            "answer": answer,
            "skill_assessed": skill_being_assessed, # Store skill for context
            "evaluation": evaluation
        }
        await save_interview_data(candidate_id, data)
        return evaluation
    except Exception as e:
        logger.error(f"Failed to evaluate candidate answer for {candidate_id}, q_id {question_id}: {e}\n{traceback.format_exc()}")
        return {}

async def get_final_report(handler: LLMHandler, candidate_id: str) -> str:
    """
    Generates the final report based on stored interview data and candidate resume,
    using the context within the LLMHandler.
    """
    data = await load_interview_data(candidate_id)
    if not data["questions"] and not data["answers"] and not data["evaluations"]:
        logger.warning(f"No interview data available for {candidate_id} to generate report.")
        return "No interview data available."
    try:
        # handler.generate_final_report() uses its internal state (resume, experience_level, key_skills, interview_history)
        report = await handler.generate_final_report()
        return report
    except Exception as e:
        logger.error(f"Failed to generate final report for {candidate_id}: {e}\n{traceback.format_exc()}")
        return "Failed to generate report."

async def get_improvement_suggestions_for_last_answer(handler: LLMHandler, candidate_id: str) -> Optional[str]:
    """
    Gets improvement suggestions for the last answered question, using the LLMHandler's context.
    Retrieves the last question and answer from the persistent data store.
    """
    data = await load_interview_data(candidate_id)
    question_ids = list(data["questions"].keys())
    
    if not question_ids: 
        logger.warning(f"No questions asked for {candidate_id} to get improvement suggestions.")
        return None
    
    last_question_id = question_ids[-1]
    # Ensure answer exists for the last question before trying to retrieve
    if last_question_id not in data["answers"]:
        logger.warning(f"No answer found for the last question {last_question_id} for {candidate_id} to provide suggestions.")
        return None

    last_question = data["questions"][last_question_id]
    last_answer = data["answers"][last_question_id]

    try:
        # handler.get_improvement_suggestions() uses its internal state (chosen_interview_level, resume, candidate_actual_experience)
        return await handler.get_improvement_suggestions(last_question, last_answer)
    except Exception as e:
        logger.error(f"Failed to get improvement suggestions for {candidate_id}, q_id {last_question_id}: {e}\n{traceback.format_exc()}")
        return "Failed to get suggestions."

# --- Main Execution Example ---

async def main():
    """
    Demonstrates the full flow of initializing an interview,
    simulating questions and answers, evaluating, and generating reports.
    """
    # 1. Candidate Information Setup
    pdf_resume_path = os.path.join(PROJECT_ROOT, "docs", "example_resume.pdf") # Adjust path as needed
    candidate_name = "HemaSenthilMurugan" 
    candidate_uuid_val = uuid.uuid4()
    candidate_id = f"{candidate_name}_{candidate_uuid_val}"
    
    # Attempt to extract resume text from PDF
    candidate_resume_text = ""
    if os.path.exists(pdf_resume_path):
        candidate_resume_text = extract_text_from_pdf(pdf_resume_path)
        if not candidate_resume_text:
            logger.error(f"Failed to extract text from PDF: {pdf_resume_path}. Using fallback text.")
            candidate_resume_text = "HemaSenthilMurugan has 6 years of experience in Python and web development, with 2 years of experience using AWS services like EC2 and S3 for scalable solutions. Possesses strong problem-solving and leadership skills. This is a fallback resume text."
    else:
        logger.warning(f"PDF resume not found at {pdf_resume_path}. Using hardcoded fallback text.")
        candidate_resume_text = "HemaSenthilMurugan has 6 years of experience in Python and web development, with 2 years of experience using AWS services like EC2 and S3 for scalable solutions. Possesses strong problem-solving and leadership skills. This is a fallback resume text."

    # This is the *categorized experience level* for the interview complexity
    candidate_interview_level = "Advanced" 
    # This is the candidate's *actual experience in years*
    candidate_actual_experience = "6 years"

    # 2. Initialize LLMHandler
    assistant_handler = LLMHandler(
        client=openai_client,
        model=DEFAULT_MODEL
    )
    # Initialize the session with overarching instructions
    await assistant_handler.initialize_session(
        initial_instructions="You are a professional interviewer evaluating candidates solely based on their provided resume and stated experience level."
    )

    # Set interview context with all essential candidate information
    await assistant_handler.set_interview_context(
        candidate_name=candidate_name,
        resume=candidate_resume_text,
        chosen_interview_level=candidate_interview_level,
        candidate_actual_experience=candidate_actual_experience
    )

    # 3. Simulate Interview Turns
    print(f"\n--- Starting Interview for {candidate_name} ({candidate_interview_level}) ---")

    # Greeting
    greeting = await assistant_handler.generate_greeting()
    print(f"\nInterviewer: {greeting}")
    # Candidate's (simulated) initial response
    await assistant_handler.process_candidate_response("Thank you for having me! I'm ready to begin.")

    # Question 1: Python/Django Project
    question_id_1 = "q1_python_django_project"
    # Generate question using LLMHandler's method which uses internal context
    question1 = await assistant_handler.generate_interview_question(skill_to_assess="Python and Django Project Management")
    if question1:
        print(f"\nInterviewer: {question1}")
        answer1 = "In my previous role at Tech Solutions Inc., I led the development of a new customer management portal using Python and Django. My key contributions included designing the database schema, implementing the user authentication system, and optimizing database queries. The main challenge was integrating with a legacy CRM, which required careful planning, data mapping, and custom middleware. I addressed this by developing a robust API layer on top of the CRM."
        await assistant_handler.process_candidate_response(answer1)
        evaluation1 = await evaluate_candidate_answer(assistant_handler, candidate_id, question_id_1, question1, answer1, "Python, Django, Project Management")
        print(f"\nEvaluation for Q1: {json.dumps(evaluation1, indent=4)}")
    else:
        print("\nFailed to generate Question 1.")

    # Question 2: AWS Experience
    question_id_2 = "q2_aws_experience"
    question2 = await assistant_handler.generate_interview_question(skill_to_assess="AWS Cloud Services")
    if question2:
        print(f"\nInterviewer: {question2}")
        answer2 = "I have experience with several AWS services, including EC2 for compute, S3 for scalable object storage, and RDS for managed databases. In one project, we migrated an on-premise application to AWS, setting up auto-scaling EC2 instances behind an Application Load Balancer and storing user-uploaded content in S3. We also utilized CloudWatch for monitoring and SQS for decoupling microservices, significantly improving our system's fault tolerance and scalability."
        await assistant_handler.process_candidate_response(answer2)
        evaluation2 = await evaluate_candidate_answer(assistant_handler, candidate_id, question_id_2, question2, answer2, "AWS Cloud Services, Scalability")
        print(f"\nEvaluation for Q2: {json.dumps(evaluation2, indent=4)}")
    else:
        print("\nFailed to generate Question 2.")

    # Question 3: Problem Solving / Architectural Design
    question_id_3 = "q3_problem_solving_design"
    question3 = await assistant_handler.generate_interview_question(skill_to_assess="Architectural Design, Problem Solving")
    if question3:
        print(f"\nInterviewer: {question3}")
        answer3 = "Certainly. One significant design challenge I faced was ensuring high availability for a critical microservice. We decided to implement a redundant architecture using an active-passive setup across multiple availability zones in AWS. This involved setting up health checks, automated failover mechanisms, and ensuring data consistency across replicas. We also employed circuit breakers and bulkheads to isolate failures."
        await assistant_handler.process_candidate_response(answer3)
        evaluation3 = await evaluate_candidate_answer(assistant_handler, candidate_id, question_id_3, question3, answer3, "High Availability, System Design, Problem Solving")
        print(f"\nEvaluation for Q3: {json.dumps(evaluation3, indent=4)}")
    else:
        print("\nFailed to generate Question 3.")


    # --- Post-Interview Actions ---

    # Generate final report
    report = await get_final_report(assistant_handler, candidate_id)
    print(f"\n--- Final Report for {candidate_name} ({candidate_interview_level} Interview) ---")
    # Using sys.stdout.buffer.write for reliable UTF-8 printing
    sys.stdout.buffer.write(report.encode('utf-8'))
    sys.stdout.buffer.write(b'\n')
    sys.stdout.flush()

    # Get improvement suggestions for the last answer
    suggestions = await get_improvement_suggestions_for_last_answer(assistant_handler, candidate_id)
    if suggestions:
        print(f"\n--- Improvement Suggestions for {candidate_name}'s Last Answer ---")
        print(suggestions)
    else:
        print("\nNo improvement suggestions available for the last answer.")

    # Generate closing statement
    closing_statement = await assistant_handler.generate_closing_statement()
    print(f"\nInterviewer: {closing_statement}")

    # End the LLM session
    await assistant_handler.end_interview_session()
    print("\n--- Interview Session Ended ---")

if __name__ == "__main__":
    asyncio.run(main())