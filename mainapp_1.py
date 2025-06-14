import asyncio
import uuid
import logging
import os
import sys
from typing import List, Optional, Dict

# === Setup Project Path (adjust if needed) ===
# This ensures that imports from other project directories work correctly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import necessary configurations from config.py
from utils.config import DEFAULT_MODEL, client as openai_client, TOTAL_INTERVIEW_DURATION, QUESTION_LIMIT, MAX_CONSECUTIVE_SILENCES, TIME_PER_QUESTION, SILENCE_TIMEOUT

# === Imports ===
# Audio processing
from audio.tts import TTS
from audio.stt import STT

# LLM handling and prompts
from llm.llm_task_1 import LLMHandler # Corrected import path for LLMHandler
from resources.prompts_3 import prompts # Prompts collection

# Evaluation and feedback functions (from eval_feed.py)
from llm.eval_feed_1 import evaluate_candidate_answer, get_final_report # Removed get_improvement_suggestions_for_last_answer

# PDF Extractor
from utils.pdf_extractor import extract_text_from_pdf # Ensure this is the correct function name from your utils


# === Logging Setup ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Interview Configuration ===
# Candidate details - Replace with actual data or dynamic input
CANDIDATE_NAME = "HEMA"
CANDIDATE_RESUME_PATH = "HEMA_M_AI_Django.pdf" # Adjust this path as needed
CANDIDATE_RESUME_FALLBACK_TEXT = "The candidate has professional experience and general competencies relevant to their field. Please assess general professional skills and problem-solving abilities."
CANDIDATE_ACTUAL_EXPERIENCE = "1 years" # Candidate's actual experience, used for LLM context
DEFAULT_INTERVIEW_LEVEL = "Basic" # Default interview difficulty level (Basic/Intermediate/Advanced)

# Audio response settings
RESPONSE_TIMEOUT = SILENCE_TIMEOUT # Time to wait for candidate's speech
MAX_SILENCE_RETRIES = MAX_CONSECUTIVE_SILENCES # Max attempts for silent responses

async def get_candidate_response(stt_processor: STT, tts_manager: TTS, retry_count: int = 0) -> Optional[str]:
    """
    Records audio and transcribes the candidate's response.
    Handles silent or unclear responses by prompting the user to try again based on configuration.
    Returns the transcription string if successful, or None if no valid response after retries.
    """
    logger.info(f"Attempting to get candidate response (Attempt {retry_count + 1})...")

    # Record audio for the configured duration
    audio_path = stt_processor.record_audio(duration=RESPONSE_TIMEOUT)

    if not audio_path:
        logger.error("Failed to record audio due to microphone issue or other recording error.")
        await tts_manager.text_to_speech("I'm sorry, I'm having trouble accessing the microphone. Please check your audio settings.")
        return None

    # Transcribe the recorded audio. The STT class now returns a dictionary.
    transcription_result = await stt_processor.transcribe_audio(audio_path)
    os.remove(audio_path) # Clean up the temporary audio file immediately

    # Check the status from the STT result
    if transcription_result["status"] == "success":
        transcribed_text = transcription_result['text'].strip()
        if transcribed_text: # Ensure text is not just whitespace
            logger.info(f"Transcription successful: '{transcribed_text}'")
            return transcribed_text
        else:
            # If status is success but text is empty/whitespace, treat as no speech
            logger.warning("STT reported success but returned empty or whitespace text. Treating as no speech.")
            status_message = "I couldn't detect any clear speech. Please try again."
            return await _handle_audio_issue_retry(stt_processor, tts_manager, retry_count, status_message)

    elif transcription_result["status"] in ["silent", "no_speech", "silent_or_no_speech"]:
        logger.warning(f"Audio issue detected: {transcription_result['message']} (Attempt {retry_count + 1}/{MAX_SILENCE_RETRIES + 1}).") # Adjust message to show total attempts
        # Pass the message directly from STT for specific feedback
        return await _handle_audio_issue_retry(stt_processor, tts_manager, retry_count, transcription_result['message'])

    elif transcription_result["status"] == "error":
        logger.error(f"Critical STT error: {transcription_result['message']}. Cannot proceed with transcription.")
        await tts_manager.text_to_speech(f"I encountered a critical error during transcription: {transcription_result['message']}. Moving on.")
        return None
    
    # Fallback return (should ideally not be reached if all statuses are handled)
    return None

async def _handle_audio_issue_retry(stt_processor: STT, tts_manager: TTS, current_retry_count: int, message: str) -> Optional[str]:
    """Helper function to manage retry logic for audio issues."""
    # MAX_SILENCE_RETRIES is the number of *additional* retries after the initial attempt.
    # So, total attempts = 1 (initial) + MAX_SILENCE_RETRIES.
    if current_retry_count < MAX_SILENCE_RETRIES:
        # Inform the user and retry
        await tts_manager.text_to_speech(f"{message} Could you please try again?")
        return await get_candidate_response(stt_processor, tts_manager, current_retry_count + 1)
    else:
        # Max retries reached
        await tts_manager.text_to_speech("We seem to be having persistent trouble with the audio. Moving on.")
        logger.info("Max audio retries reached. Moving on from current question.")
        return None

async def main():
    candidate_id = f"{CANDIDATE_NAME.replace(' ', '_')}_{uuid.uuid4()}"
    
    # === 1. Extract Resume Text ===
    logger.info(f"Attempting to extract resume text from: {CANDIDATE_RESUME_PATH}")
    extracted_resume_text = await extract_text_from_pdf(CANDIDATE_RESUME_PATH)
    
    if not extracted_resume_text or len(extracted_resume_text.strip()) < 50: # Check for minimal content
        logger.warning(f"Could not extract sufficient resume text from '{CANDIDATE_RESUME_PATH}'. Using fallback text.")
        candidate_resume_final_text = CANDIDATE_RESUME_FALLBACK_TEXT
    else:
        candidate_resume_final_text = extracted_resume_text
        logger.info("Resume text extracted successfully.")
    
    # === 2. Initialize LLM Handler and Audio Processors ===
    assistant_handler = LLMHandler(
        client=openai_client,
        model=DEFAULT_MODEL
    )
    # Initialize the LLMHandler with system instructions and interview context
    await assistant_handler.initialize_session(
        initial_instructions="You are a professional technical interviewer for a software development role. Evaluate candidates based on their resume and stated experience."
    )
    # Set the core interview context
    await assistant_handler.set_interview_context(
        candidate_name=CANDIDATE_NAME,
        resume=candidate_resume_final_text,
        chosen_interview_level=DEFAULT_INTERVIEW_LEVEL,
        candidate_actual_experience=CANDIDATE_ACTUAL_EXPERIENCE
    )

    tts_manager = TTS()
    stt_processor = STT()

    # Variables for interview flow
    asked_questions = []
    last_answer = ""
    interview_history = [] # For logging/displaying overall interview flow
    consecutive_silence_count = 0

    logger.info(f"Starting interview for candidate: {CANDIDATE_NAME} (ID: {candidate_id})")
    logger.info(f"Configured for ~{TOTAL_INTERVIEW_DURATION / 60:.0f} minute interview with up to {QUESTION_LIMIT} questions.")

    # === 3. Interview Greeting ===
    # Using LLMHandler's method to generate the greeting
    greeting = await assistant_handler.generate_greeting()
    await tts_manager.text_to_speech(greeting)
    interview_history.append(f"Interviewer: {greeting}")

    # === 4. Ask for Self-Introduction ===
    self_intro_question = "Please tell me a little about yourself and your background."
    await tts_manager.text_to_speech(self_intro_question)
    interview_history.append(f"Interviewer: {self_intro_question}")
    asked_questions.append(self_intro_question) # Keep track of asked questions

    # === 5. Get Candidate's Self-Introduction Response ===
    candidate_response = await get_candidate_response(stt_processor, tts_manager)
    if candidate_response:
        print(f"\nCandidate: {candidate_response}")
        await assistant_handler.process_candidate_response(candidate_response) # Add to LLM history
        interview_history.append(f"Candidate: {candidate_response}")
        last_answer = candidate_response
        consecutive_silence_count = 0
    else:
        last_answer = ""
        logger.warning("No valid self-introduction received. Consecutive silence counter increased.")
        consecutive_silence_count += 1
        # If no intro, we might prompt again or just proceed. The get_candidate_response handles retries now.

    # === 6. Evaluate Self-Introduction (Optional) ===
    if last_answer:
        # Evaluate using the eval_feed function, which uses LLMHandler internally
        evaluation = await evaluate_candidate_answer(
            assistant_handler,
            candidate_id,
            "self_introduction", # Unique ID for this question
            self_intro_question,
            last_answer,
            "Communication Skills" # Skill assessed for self-introduction
        )
        logger.info(f"Evaluation for self-introduction: {evaluation.get('overall_score', 'N/A')}")
        # print(f"\nSelf-Intro Evaluation: {json.dumps(evaluation, indent=4)}") # Uncomment for verbose output


    # === 7. Main Interview Loop for Subsequent Questions ===
    for i in range(QUESTION_LIMIT):
        question_count = i + 1 # Start from 1
        logger.info(f"--- Question {question_count} ---")

        current_question = ""
        # Logic to determine the type of question (resume-based, follow-up, dynamic)
        # We'll use LLMHandler's general question generation which leverages context
        # and its internal prompts to decide.
        if i == 0: # First question after intro
            # This is effectively a "resume-based" question
            current_question = await assistant_handler.generate_interview_question(
                skill_to_assess="resume background, core competencies"
            )
        elif last_answer and len(asked_questions) > 0 and (question_count -1) % 2 == 1:
            # Attempt a follow-up question, relying on LLMHandler's ability to create one
            # We explicitly format the follow-up prompt and pass context to LLMHandler's task method
            prev_q_content = asked_questions[-1]
            follow_up_prompt_content = prompts.follow_up_prompt.format(
                previous_question=prev_q_content,
                candidate_answer=last_answer,
                candidate_actual_experience=assistant_handler.candidate_actual_experience,
                chosen_interview_level=assistant_handler.chosen_interview_level,
                key_skills=", ".join(assistant_handler.key_skills) if assistant_handler.key_skills else "N/A"
            )
            current_question = await assistant_handler.get_llm_response_for_task(
                prompt_content=follow_up_prompt_content,
                system_instructions="You are an expert interviewer. Generate a concise follow-up question based on the previous answer and interview context.",
                temperature=0.7
            )
        else:
            # Generate a general dynamic question based on current history and context
            # We explicitly format the dynamic prompt and pass context to LLMHandler's task method
            dynamic_prompt_content = prompts.resume_interview_prompt.format(
                resume=assistant_handler.candidate_resume,
                last_answer=last_answer,
                asked_questions=", ".join(asked_questions),
                question_count=len(asked_questions) + 1, # Next question count
                candidate_actual_experience=assistant_handler.candidate_actual_experience,
                chosen_interview_level=assistant_handler.chosen_interview_level,
                key_skills=", ".join(assistant_handler.key_skills) if assistant_handler.key_skills else "N/A"
            )
            current_question = await assistant_handler.get_llm_response_for_task(
                prompt_content=dynamic_prompt_content,
                system_instructions="You are an expert interviewer. Generate a dynamic interview question tailored to the candidate's resume and the interview flow.",
                temperature=0.7
            )

        if not current_question:
            logger.error(f"Failed to generate question {question_count}. Skipping this turn.")
            continue # Skip to next loop iteration if question generation fails

        await tts_manager.text_to_speech(current_question)
        interview_history.append(f"Interviewer: {current_question}")
        asked_questions.append(current_question) # Add to list of asked questions

        candidate_response = await get_candidate_response(stt_processor, tts_manager)
        if candidate_response:
            print(f"\nCandidate: {candidate_response}")
            await assistant_handler.process_candidate_response(candidate_response) # Add to LLM history
            interview_history.append(f"Candidate: {candidate_response}")
            last_answer = candidate_response
            consecutive_silence_count = 0
        else:
            last_answer = "" # Clear last answer if no response
            logger.warning(f"No valid response received for question {question_count}. Consecutive silence counter increased.")
            consecutive_silence_count += 1
            if consecutive_silence_count >= MAX_SILENCE_RETRIES:
                await tts_manager.text_to_speech("It seems we are having persistent audio issues. Let's proceed to the next question.")
                consecutive_silence_count = 0 # Reset counter after warning
                continue # Move to next question without evaluation/feedback for this turn

        # === 8. Evaluate Candidate Answer (Internal only, no AI speech feedback) ===
        if last_answer and current_question: # Only evaluate if there was an answer
            # Attempt to identify the skill being assessed (can be refined using LLMHandler's identify_skill)
            skill_for_eval = "General Professional Competency" # Renamed from "General Technical Competency" for broader applicability

            if "python" in current_question.lower() or "llm" in current_question.lower() or "rag" in current_question.lower() or "machine learning" in current_question.lower() or "artificial intelligence" in current_question.lower():
                skill_for_eval = "AI/ML Proficiency"
            elif "aws" in current_question.lower() or "cloud" in current_question.lower() or "azure" in current_question.lower() or "gcp" in current_question.lower() or "devops" in current_question.lower():
                skill_for_eval = "Cloud Computing & DevOps"
            elif "architecture" in current_question.lower() or "design" in current_question.lower() or "scalable" in current_question.lower() or "system" in current_question.lower():
                skill_for_eval = "System Design & Architecture"
            elif "project" in current_question.lower() or "team" in current_question.lower() or "collaboration" in current_question.lower() or "agile" in current_question.lower() or "scrum" in current_question.lower():
                skill_for_eval = "Project Management & Collaboration"
            elif "challenges" in current_question.lower() or "problem" in current_question.lower() or "resolve" in current_question.lower() or "solution" in current_question.lower():
                skill_for_eval = "Problem Solving"
        # === New Domains ===
            elif "marketing" in current_question.lower() or "campaign" in current_question.lower() or "seo" in current_question.lower() or "content" in current_question.lower() or "branding" in current_question.lower():
                skill_for_eval = "Marketing & Brand Management"
            elif "sales" in current_question.lower() or "client" in current_question.lower() or "customer" in current_question.lower() or "negotiation" in current_question.lower() or "revenue" in current_question.lower() or "pipeline" in current_question.lower():
                skill_for_eval = "Sales & Client Relations"
            elif "hr" in current_question.lower() or "human resources" in current_question.lower() or "recruitment" in current_question.lower() or "employee" in current_question.lower() or "onboarding" in current_question.lower() or "benefits" in current_question.lower():
                skill_for_eval = "Human Resources Management"
            elif "teach" in current_question.lower() or "educat" in current_question.lower() or "curriculum" in current_question.lower() or "student" in current_question.lower() or "lesson" in current_question.lower():
                skill_for_eval = "Teaching & Pedagogy"
            elif "communication" in current_question.lower() or "present" in current_question.lower() or "explain" in current_question.lower() or "interpersonal" in current_question.lower():
                skill_for_eval = "Communication Skills"
        # Common business/general skills that might not fit specific domains above
            elif "strategy" in current_question.lower() or "business" in current_question.lower() or "market" in current_question.lower() or "operations" in current_question.lower():
                skill_for_eval = "Business Acumen & Strategy"
            elif "data" in current_question.lower() or "analyze" in current_question.lower() or "report" in current_question.lower() or "metrics" in current_question.lower():
                skill_for_eval = "Data Analysis & Interpretation"
            
            # Evaluate the answer using the eval_feed function (which uses LLMHandler)
            evaluation = await evaluate_candidate_answer(
                assistant_handler,
                candidate_id,
                f"question_{question_count}", # Unique ID for this question
                current_question,
                last_answer,
                skill_for_eval # Identified skill
            )
            logger.info(f"Evaluation for question {question_count}: Overall Score = {evaluation.get('overall_score', 'N/A')}")
            # print(f"\nEvaluation for Q{question_count}: {json.dumps(evaluation, indent=4)}") # Uncomment for verbose output

            # Removed: No AI speech feedback on improvement suggestions after each answer.
            # get_improvement_suggestions_for_last_answer is no longer used.


    # === 9. Closing Statement ===
    # Using LLMHandler's method to generate the closing statement
    closing_statement = await assistant_handler.generate_closing_statement()
    await tts_manager.text_to_speech(closing_statement)
    interview_history.append(f"Interviewer: {closing_statement}")

    # --- Candidate Q&A Session ---
    # AI bot asks if the candidate has questions and provides a skip option
    await tts_manager.text_to_speech("Do you have any questions for me? You can ask up to two questions. Or, if you have no questions, you can say 'skip'.")
    interview_history.append(f"Interviewer: Do you have any questions for me? You can ask up to two questions. Or, if you have no questions, you can say 'skip'.")

    for q_num in range(1, 3):  # Loop for a maximum of 2 questions
        await tts_manager.text_to_speech(f"Please ask your question number {q_num}. Or, if you're done, just say 'skip'.")
        
        # Give time for the candidate to ask their question or say 'skip'
        candidate_response = await get_candidate_response(stt_processor, tts_manager)

        if candidate_response:
            # Check for 'skip' command (case-insensitive)
            if "skip" in candidate_response.lower():
                await tts_manager.text_to_speech("Understood. No problem at all.")
                interview_history.append(f"Candidate: Skip")
                interview_history.append(f"Interviewer: Understood. No problem at all.")
                break # Exit the Q&A loop if candidate says 'skip'
            
            print(f"\nCandidate's question: {candidate_response}")
            interview_history.append(f"Candidate's question {q_num}: {candidate_response}")

            # Add candidate's question to LLM's history for context
            await assistant_handler.process_candidate_response(f"Candidate asks: {candidate_response}")

            # Get LLM to answer the candidate's question, maintaining an interviewer persona
            ai_answer = await assistant_handler.get_next_assistant_response(
                additional_instructions="Please answer the candidate's question clearly, concisely, and politely, maintaining the role of an interviewer. Avoid being overly formal or too brief."
            )
            
            if ai_answer:
                await tts_manager.text_to_speech(ai_answer)
                interview_history.append(f"Interviewer's Answer: {ai_answer}")
            else:
                # Fallback polite response if LLM fails to generate an answer
                await tts_manager.text_to_speech("I apologize, I didn't quite catch that question or formulate an answer. Could you please rephrase?")
                interview_history.append(f"Interviewer's Answer: I apologize, I didn't quite catch that question or formulate an answer.")

        else:
            # If no question/response is detected (silence), politely acknowledge and break
            await tts_manager.text_to_speech("It seems you don't have a question at this moment.")
            interview_history.append(f"Interviewer: It seems you don't have a question at this moment.")
            break  # Exit loop if no question is asked

        # If it's the first question and not skipped, prompt for another one
        if q_num == 1:
            await tts_manager.text_to_speech("Do you have any other questions? Or just say 'skip' if you're done.")
            interview_history.append(f"Interviewer: Do you have any other questions? Or just say 'skip' if you're done.")
            # No brief silence check here; the next loop iteration will handle the response or explicit 'skip'.

    # Final thank you after candidate questions (or skip)
    await tts_manager.text_to_speech("Thank thank you for your questions. This concludes our interview.")
    interview_history.append(f"Interviewer: Thank you for your questions. This concludes our interview.")
    # --- End Candidate Q&A Session ---

    # === 10. Final Report ===
    try:
        final_report = await get_final_report(assistant_handler, candidate_id)
        print("\n--- Final Interview Report ---")
        # Use sys.stdout.buffer.write for robust UTF-8 printing, especially for complex LLM outputs
        sys.stdout.buffer.write(final_report.encode('utf-8'))
        sys.stdout.buffer.write(b'\n')
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"An unexpected error occurred during final report generation: {e}")
        print(f"Failed to generate final report due to an unexpected error.")

    # === 11. Cleanup ===
    await assistant_handler.end_interview_session() # Clear LLMHandler's internal state
    await tts_manager.shutdown() # Ensure TTS resources are released
    logger.info(f"Interview session for {CANDIDATE_NAME} (ID: {candidate_id}) concluded.")

if __name__ == "__main__":
    asyncio.run(main())