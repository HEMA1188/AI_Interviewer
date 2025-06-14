import asyncio
import json
import logging
import os
import sys
from typing import List, Optional, Dict

# === Setup Project Path (adjust if needed) ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === Imports ===
# Assuming llm_task is the correct path to your LLMHandler and call_openai
from llm.llm_task_1 import call_openai, DEFAULT_MODEL, LLMHandler # Renamed from AssistantLLMHandler
from resources.prompts_3 import prompts

# === Logging Setup ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def generate_follow_up_question(previous_question: str, candidate_answer: str, model: str = DEFAULT_MODEL) -> str:
    """Generates a follow-up question using the standard OpenAI API."""
    try:
        response = await call_openai(prompts.follow_up_prompt.format(
            previous_question=previous_question, candidate_answer=candidate_answer
        ), model=model)
        logger.info(f"Follow-up question generated for: {previous_question}")
        return response
    except Exception as e:
        logger.error(f"Error generating follow-up question: {e}")
        return "Sorry, I couldn't generate a follow-up question at the moment."

async def generate_resume_based_question(resume: str, asked_questions: List[str], model: str = DEFAULT_MODEL) -> str:
    """Generates a resume-based question using the standard OpenAI API."""
    try:
        response = await call_openai(prompts.resume_based_question_prompt.format(
            resume=resume, asked_questions=", ".join(asked_questions)
        ), model=model)
        logger.info("Resume-based question generated.")
        return response
    except Exception as e:
        logger.error(f"Error generating resume-based question: {e}")
        return "Sorry, I couldn't generate a resume-based question at the moment."

async def generate_dynamic_question(resume: str, last_answer: str, asked_questions: List[str], question_count: int, model: str = DEFAULT_MODEL) -> str:
    """Generates a dynamic interview question using the standard OpenAI API, based on the resume."""
    try:
        response = await call_openai(prompts.resume_interview_prompt.format(
            resume=resume, last_answer=last_answer, asked_questions=", ".join(asked_questions), question_count=question_count
        ), model=model)
        logger.info(f"Dynamic question generated after {question_count} questions asked, based on resume.")
        return response
    except Exception as e:
        logger.error(f"Error generating dynamic question: {e}")
        return "Sorry, I couldn't generate a dynamic interview question at the moment."

async def generate_greeting(candidate_name: str, resume: str, model: str = DEFAULT_MODEL) -> str:
    """Generates a warm-up greeting using the standard OpenAI API."""
    try:
        response = await call_openai(prompts.greet_warmup_prompt.format(
            candidate_name=candidate_name, resume=resume
        ), model=model)
        logger.info(f"Greeting generated for candidate: {candidate_name}")
        return response
    except Exception as e:
        logger.error(f"Error generating greeting: {e}")
        return f"Hello {candidate_name}, welcome to the interview."

async def generate_closing(model: str = DEFAULT_MODEL) -> str:
    """Generates the closing statement using the standard OpenAI API."""
    try:
        response = await call_openai(prompts.closing_prompt, model=model)
        logger.info("Closing statement generated.")
        return response
    except Exception as e:
        logger.error(f"Error generating closing statement: {e}")
        return "Thank you for your time. We will get back to you soon."

async def generate_all_questions(candidate_name: str, resume: str, asked_questions: List[str], question_count: int, last_answer: str, model: str = DEFAULT_MODEL):
    """Generate all questions (greeting, dynamic, follow-up, resume-based) for an interview round."""
    try:
        # Greeting
        greeting = await generate_greeting(candidate_name, resume, model)

        # Dynamic Question
        dynamic_question = await generate_dynamic_question(resume, last_answer, asked_questions, question_count, model)

        # Follow-up Question
        follow_up_question = await generate_follow_up_question(dynamic_question, last_answer, model)

        # Resume-based Question
        resume_question = await generate_resume_based_question(resume, asked_questions, model)

        # Closing Statement
        closing = await generate_closing(model)

        return {
            "greeting": greeting,
            "dynamic_question": dynamic_question,
            "follow_up_question": follow_up_question,
            "resume_question": resume_question,
            "closing": closing
        }
    except Exception as e:
        logger.error(f"Error in generating all questions: {e}")
        return {}