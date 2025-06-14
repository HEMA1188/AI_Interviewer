import asyncio
import uuid
import logging
import os
import sys
import json # <--- ADDED: Import for JSON parsing
from typing import List, Optional, Dict

# === Setup Project Path (adjust if needed) ===
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
from llm.llm_task_1 import LLMHandler
from resources.prompts_3 import prompts # Prompts collection

# Evaluation and feedback functions (from eval_feed.py)
from llm.eval_feed_1 import evaluate_candidate_answer, get_final_report

# PDF Extractor
from utils.pdf_extractor import extract_text_from_pdf

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
    Handles silent responses by prompting the user to try again.
    Returns the transcription or None if no valid response after max retries.
    """
    audio_path = stt_processor.record_audio(duration=RESPONSE_TIMEOUT)

    if not audio_path:
        logger.error("Failed to record audio. Check microphone setup.")
        return None

    transcription_result = await stt_processor.transcribe_audio(audio_path)
    os.remove(audio_path)

    if transcription_result["status"] == "success":
        logger.info(f"Transcription: {transcription_result['text']}")
        return transcription_result["text"]
    elif transcription_result["status"] in ["silent", "no_speech", "silent_or_no_speech"]:
        logger.warning(f"Audio issue detected: {transcription_result['message']} (Retry {retry_count + 1}/{MAX_SILENCE_RETRIES}).")
        
        if retry_count < MAX_SILENCE_RETRIES:
            await tts_manager.text_to_speech(f"{transcription_result['message']} Could you please try again?")
            return await get_candidate_response(stt_processor, tts_manager, retry_count + 1)
        else:
            await tts_manager.text_to_speech("We seem to be having persistent trouble with the audio. Moving on.")
            return None
    elif transcription_result["status"] == "error":
        logger.error(f"STT error: {transcription_result['message']}. Returning None.")
        return None
    
    return None

async def main():
    candidate_id = f"{CANDIDATE_NAME.replace(' ', '_')}_{uuid.uuid4()}"
    
    # === 1. Extract Resume Text ===
    logger.info(f"Attempting to extract resume text from: {CANDIDATE_RESUME_PATH}")
    extracted_resume_text = await extract_text_from_pdf(CANDIDATE_RESUME_PATH) # <-- AWAITED here
    
    if not extracted_resume_text or len(extracted_resume_text.strip()) < 50:
        logger.warning(f"Could not extract sufficient resume text from '{CANDIDATE_RESUME_PATH}'.")
    # Prompt the user for manual input if resume extraction fails
        print("\n[IMPORTANT]: Resume text extraction failed or was insufficient.")
        manual_resume_summary = input("Please provide a brief summary of the candidate's background and key skills (e.g., '5 years experience in digital marketing, specializing in SEO and content creation'):\n> ")
        if manual_resume_summary.strip():
            candidate_resume_final_text = manual_resume_summary
            logger.info("Using manually provided resume summary.")
        else:
            candidate_resume_final_text = "Candidate has general professional experience and relevant skills in their chosen field." # Default generic if user provides nothing
            logger.warning("No manual summary provided. Using generic fallback.")
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
        initial_instructions="You are a professional interviewer. Your role is to assess the candidate's suitability for a position based on their resume and stated experience, focusing on relevant skills and competencies."
    )
    
    # Set the core interview context
    await assistant_handler.set_interview_context(
        candidate_name=CANDIDATE_NAME,
        resume=candidate_resume_final_text,
        chosen_interview_level=DEFAULT_INTERVIEW_LEVEL,
        candidate_actual_experience=CANDIDATE_ACTUAL_EXPERIENCE
    )

    # --- NEW: Extract Key Skills from Resume using LLM ---
    try:
        skill_extraction_prompt_content = prompts.EXTRACT_SKILLS_PROMPT.format(resume=candidate_resume_final_text)
        raw_skills_json = await assistant_handler.get_llm_response_for_task(
            prompt_content=skill_extraction_prompt_content,
            system_instructions="You are an expert at identifying core professional skills and competencies from resumes.", # Updated system instruction
            temperature=0.0
        )
        
        skills_data = json.loads(raw_skills_json)
        extracted_key_skills = skills_data.get("skills", [])
        
        assistant_handler.key_skills = extracted_key_skills
        logger.info(f"Extracted Key Skills: {', '.join(extracted_key_skills)}")

    except Exception as e:
        logger.error(f"Failed to extract key skills from resume: {e}. Proceeding with general questioning.")
        assistant_handler.key_skills = [] # Fallback if extraction fails

    tts_manager = TTS()
    stt_processor = STT()

    # Variables for interview flow
    asked_questions = []
    last_answer = ""
    interview_history = []
    consecutive_silence_count = 0

    logger.info(f"Starting interview for candidate: {CANDIDATE_NAME} (ID: {candidate_id})")
    logger.info(f"Configured for ~{TOTAL_INTERVIEW_DURATION / 60:.0f} minute interview with up to {QUESTION_LIMIT} questions.")

    # === 3. Interview Greeting ===
    greeting = await assistant_handler.generate_greeting()
    await tts_manager.text_to_speech(greeting)
    interview_history.append(f"Interviewer: {greeting}")

    # === 4. Ask for Self-Introduction ===
    self_intro_question = "Please tell me a little about yourself and your background."
    await tts_manager.text_to_speech(self_intro_question)
    interview_history.append(f"Interviewer: {self_intro_question}")
    asked_questions.append(self_intro_question)

    # === 5. Get Candidate's Self-Introduction Response ===
    candidate_response = await get_candidate_response(stt_processor, tts_manager)
    if candidate_response:
        print(f"\nCandidate: {candidate_response}")
        await assistant_handler.process_candidate_response(candidate_response)
        interview_history.append(f"Candidate: {candidate_response}")
        last_answer = candidate_response
        consecutive_silence_count = 0
    else:
        last_answer = ""
        logger.warning("No valid self-introduction received. Consecutive silence counter increased.")
        consecutive_silence_count += 1

    # === 6. Evaluate Self-Introduction (Optional) ===
    if last_answer:
        evaluation = await evaluate_candidate_answer(
            assistant_handler,
            candidate_id,
            "self_introduction",
            self_intro_question,
            last_answer,
            "Communication Skills" # Specific skill for self-intro
        )
        logger.info(f"Evaluation for self-introduction: {evaluation.get('overall_score', 'N/A')}")

    # === 7. Main Interview Loop for Subsequent Questions ===
    for i in range(QUESTION_LIMIT):
        question_count = i + 1
        logger.info(f"--- Question {question_count} ---")

        current_question = ""
        # Format key skills for the prompt string
        key_skills_formatted = ", ".join(assistant_handler.key_skills) if assistant_handler.key_skills else "general professional competencies"

        if i == 0: # First question after intro
            current_question = await assistant_handler.generate_interview_question(
                skill_to_assess="core professional skills relevant to resume",
                #key_skills=key_skills_formatted # Pass formatted key skills
            )
        elif last_answer and len(asked_questions) > 0 and (question_count - 1) % 2 == 1:
            prev_q_content = asked_questions[-1]
            follow_up_prompt_content = prompts.follow_up_prompt.format(
                previous_question=prev_q_content,
                candidate_answer=last_answer,
                candidate_actual_experience=assistant_handler.candidate_actual_experience,
                chosen_interview_level=assistant_handler.chosen_interview_level,
                key_skills=key_skills_formatted # Pass formatted key skills
            )
            current_question = await assistant_handler.get_llm_response_for_task(
                prompt_content=follow_up_prompt_content,
                system_instructions="You are an expert interviewer. Generate a concise follow-up question based on the previous answer and interview context.",
                temperature=0.7
            )
        else:
            dynamic_prompt_content = prompts.resume_interview_prompt.format(
                resume=assistant_handler.candidate_resume,
                last_answer=last_answer,
                asked_questions=", ".join(asked_questions),
                question_count=len(asked_questions) + 1,
                candidate_actual_experience=assistant_handler.candidate_actual_experience,
                chosen_interview_level=assistant_handler.chosen_interview_level,
                key_skills=key_skills_formatted # Pass formatted key skills
            )
            current_question = await assistant_handler.get_llm_response_for_task(
                prompt_content=dynamic_prompt_content,
                system_instructions="You are an expert interviewer. Generate a dynamic interview question tailored to the candidate's resume and the interview flow.",
                temperature=0.7
            )

        if not current_question:
            logger.error(f"Failed to generate question {question_count}. Skipping this turn.")
            continue

        await tts_manager.text_to_speech(current_question)
        interview_history.append(f"Interviewer: {current_question}")
        asked_questions.append(current_question)

        candidate_response = await get_candidate_response(stt_processor, tts_manager)
        if candidate_response:
            print(f"\nCandidate: {candidate_response}")
            await assistant_handler.process_candidate_response(candidate_response)
            interview_history.append(f"Candidate: {candidate_response}")
            last_answer = candidate_response
            consecutive_silence_count = 0
        else:
            last_answer = ""
            logger.warning(f"No valid response received for question {question_count}. Consecutive silence counter increased.")
            consecutive_silence_count += 1
            if consecutive_silence_count >= MAX_SILENCE_RETRIES:
                await tts_manager.text_to_speech("It seems we are having persistent audio issues. Let's proceed to the next question.")
                consecutive_silence_count = 0
                continue

        # === 8. Evaluate Candidate Answer (Internal only, no AI speech feedback) ===
        if last_answer and current_question:
            # LLM-driven identification of the skill being assessed for this question
            try:
                skill_identification_prompt = prompts.IDENTIFY_SKILL_PROMPT.format(
                    resume=assistant_handler.candidate_resume, # Pass resume for context
                    question=current_question
                )
                
                identified_skill = await assistant_handler.get_llm_response_for_task(
                    prompt_content=skill_identification_prompt,
                    system_instructions="You are an expert in skill taxonomy. Based on the interview question and resume, identify the single most relevant skill being assessed. Provide only the skill name.",
                    temperature=0.0
                )
                
                # Clean up the identified skill
                skill_for_eval = identified_skill.strip().replace('.', '').replace('"', '')
                if not skill_for_eval: # Fallback if LLM returns empty or malformed string
                    skill_for_eval = "General Professional Competency"
                
            except Exception as e:
                logger.error(f"Error identifying skill for question: {current_question}. Defaulting to 'General Professional Competency'. Error: {e}")
                skill_for_eval = "General Professional Competency"

            logger.info(f"Question '{current_question[:50]}...' is assessing skill: {skill_for_eval}")
            
            evaluation = await evaluate_candidate_answer(
                assistant_handler,
                candidate_id,
                f"question_{question_count}",
                current_question,
                last_answer,
                skill_for_eval # Dynamically identified skill
            )
            logger.info(f"Evaluation for question {question_count}: Overall Score = {evaluation.get('overall_score', 'N/A')}")

    # === 9. Closing Statement ===
    closing_statement = await assistant_handler.generate_closing_statement()
    await tts_manager.text_to_speech(closing_statement)
    interview_history.append(f"Interviewer: {closing_statement}")

    # --- Candidate Q&A Session ---
    await tts_manager.text_to_speech("Do you have any questions for me? You can ask up to two questions. Or, if you have no questions, you can say 'skip'.")
    interview_history.append(f"Interviewer: Do you have any questions for me? You can ask up to two questions. Or, if you have no questions, you can say 'skip'.")

    for q_num in range(1, 3):
        await tts_manager.text_to_speech(f"Please ask your question number {q_num}. Or, if you're done, just say 'skip'.")
        
        candidate_response = await get_candidate_response(stt_processor, tts_manager)

        if candidate_response:
            if "skip" in candidate_response.lower():
                await tts_manager.text_to_speech("Understood. No problem at all.")
                interview_history.append(f"Candidate: Skip")
                interview_history.append(f"Interviewer: Understood. No problem at all.")
                break
            
            print(f"\nCandidate's question: {candidate_response}")
            interview_history.append(f"Candidate's question {q_num}: {candidate_response}")

            await assistant_handler.process_candidate_response(f"Candidate asks: {candidate_response}")

            ai_answer = await assistant_handler.get_next_assistant_response(
                additional_instructions="Please answer the candidate's question clearly, concisely, and politely, maintaining the role of an interviewer. Avoid being overly formal or too brief."
            )
            
            if ai_answer:
                await tts_manager.text_to_speech(ai_answer)
                interview_history.append(f"Interviewer's Answer: {ai_answer}")
            else:
                await tts_manager.text_to_speech("I apologize, I didn't quite catch that question or formulate an answer. Could you please rephrase?")
                interview_history.append(f"Interviewer's Answer: I apologize, I didn't quite catch that question or formulate an answer.")

        else:
            await tts_manager.text_to_speech("It seems you don't have a question at this moment.")
            interview_history.append(f"Interviewer: It seems you don't have a question at this moment.")
            break

    await tts_manager.text_to_speech("Thank you for your questions. This concludes our interview.")
    interview_history.append(f"Interviewer: Thank you for your questions. This concludes our interview.")

    # === 10. Final Report ===
    # --- START DEBUGGING BLOCK ---
    logger.info("--- Preparing for Final Report Generation ---")
    logger.info(f"LLMHandler.candidate_resume set: {bool(assistant_handler.candidate_resume)}")
    if assistant_handler.candidate_resume:
        logger.info(f"Resume snippet: {assistant_handler.candidate_resume[:100]}...")
    logger.info(f"LLMHandler.interview_history length: {len(assistant_handler.interview_history)}")
    logger.info(f"LLMHandler.key_skills set: {bool(assistant_handler.key_skills)}")
    if assistant_handler.key_skills:
        logger.info(f"Key skills: {', '.join(assistant_handler.key_skills)}")
    logger.info(f"LLMHandler.chosen_interview_level set: {bool(assistant_handler.chosen_interview_level)}")
    logger.info(f"LLMHandler.candidate_actual_experience set: {bool(assistant_handler.candidate_actual_experience)}")
    logger.info("--- End Debugging Block ---")

    try:
        final_report = await get_final_report(assistant_handler, candidate_id)
        print("\n--- Final Interview Report ---")
        sys.stdout.buffer.write(final_report.encode('utf-8'))
        sys.stdout.buffer.write(b'\n')
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"An unexpected error occurred during final report generation: {e}")
        print(f"Failed to generate final report due to an unexpected error.")

    # === 11. Cleanup ===
    await assistant_handler.end_interview_session()
    await tts_manager.shutdown()
    logger.info(f"Interview session for {CANDIDATE_NAME} (ID: {candidate_id}) concluded.")

if __name__ == "__main__":
    asyncio.run(main())