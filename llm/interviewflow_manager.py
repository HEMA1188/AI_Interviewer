# interview_flow_manager.py (or could be part of your main app.py)
import asyncio
import logging
from typing import List, Optional, Dict
import json

# === Setup Project Path ===
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === Imports ===
from llm.llm_task_1 import LLMHandler # Import the LLMHandler class
from resources.prompts_3 import prompts # Import the prompts list

# === Logging Setup ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def run_interview_round(llm_handler: LLMHandler, current_question_count: int, last_candidate_answer: str) -> Dict[str, str]:
    """
    Manages a single round of the interview, generating questions using the LLMHandler.
    This function demonstrates how different question types would be generated.
    """
    generated_outputs = {}

    # Initial Greeting (usually only once at the very start)
    if current_question_count == 0: # Or some other condition to determine if it's the first interaction
        greeting = await llm_handler.generate_greeting()
        generated_outputs["greeting"] = greeting
        logger.info(f"AI: {greeting}")
        # In a real app, you'd then get candidate's first response.
    
    # Example: Generate a dynamic question
    # LLMHandler.generate_interview_question is versatile and can generate resume-based, dynamic etc.
    # We need to tell it which prompt to use and pass all the context it needs.
    # LLMHandler already stores resume, experience, level, key skills, and history.
    
    # To generate a specific type of question (e.g., dynamic, resume-based, follow-up),
    # you'd need to modify `LLMHandler.generate_interview_question` to accept a `prompt_type`
    # or create specific methods within `LLMHandler` for each.
    # For now, let's use the existing `generate_interview_question` and let it decide.

    # Example: Generate a regular interview question (which will consider resume, level, etc.)
    # The `LLMHandler.generate_interview_question` already uses `GENERATE_QUESTION_PROMPT`
    # which has logic for various complexity levels and resume focus.
    # For truly dynamic/resume-based/follow-up questions, LLMHandler would call a more
    # specific prompt from prompts_1.py, using its internal state.

    # Let's assume LLMHandler gets a method to generate a "next appropriate question"
    # which internally picks between resume_based_question_prompt, follow_up_prompt, etc.
    
    # *** IMPORTANT: The LLMHandler needs to be enhanced to decide WHICH prompt to use ***
    # For instance, a new method like:
    # `llm_handler.generate_next_interview_question(previous_question: str, candidate_answer: str, question_count: int)`
    # This method would internally decide if it's a follow-up, a new resume-based question, etc.

    # For demonstration, let's assume `generate_interview_question` is smart enough or we call specific ones.
    
    # If the last interaction was a question and a non-empty answer:
    if last_candidate_answer and last_candidate_answer != "(Silence or no audible speech detected)": # Check for actual content
        # If there's a previous question to follow up on (from history, not passed here)
        # Assuming you'd get the last question from `llm_handler.interview_history`
        # This requires a more robust history management in LLMHandler to get `last_question`
        
        # This section requires LLMHandler to hold the context of the *last question asked*.
        # Let's assume you fetch the last question from `llm_handler.interview_history`
        # For simplicity in this example, let's just generate a generic next question
        # using the existing generate_interview_question which takes internal context.
        pass # The direct follow-up generation below is problematic.

    # The existing `generate_interview_question` uses `GENERATE_QUESTION_PROMPT`.
    # To truly generate `dynamic_question`, `follow_up_question`, `resume_question` *selectively*,
    # you need LLMHandler to have more specific methods or a more intelligent master question generation method.
    
    # For now, let's stick to generating a generic "next interview question" as the core.
    # You would pass specific skills to assess if needed.
    next_question = await llm_handler.generate_interview_question()
    if next_question:
        generated_outputs["next_interview_question"] = next_question
        logger.info(f"AI: {next_question}")
    else:
        generated_outputs["next_interview_question"] = "I couldn't generate the next question."

    # Closing statement (usually at the end of the interview)
    # You'd have logic in your main interview loop to decide when to call this.
    # For example, after 'X' number of questions or upon a command.
    # For demo:
    # if current_question_count >= 5: # Example: generate closing after 5 questions
    #     closing_statement = await llm_handler.generate_closing_statement()
    #     generated_outputs["closing_statement"] = closing_statement
    #     logger.info(f"AI: {closing_statement}")

    return generated_outputs

# Example usage (in your main interview orchestrator, not in this file typically)
async def main_interview_orchestrator():
    # Assuming client is configured elsewhere as per llm_task_1.py
    from llm.llm_task_1 import client
    
    handler = LLMHandler(client=client)

    # Set up interview context first! This is crucial.
    await handler.set_interview_context(
        candidate_name="John Doe",
        resume="""John Doe has 5 years of experience in Python and Django development.
        He led a team developing a scalable web application using REST APIs and PostgreSQL.
        Strong problem-solving and communication skills. Involved in Agile methodologies.""",
        chosen_interview_level="Intermediate",
        candidate_actual_experience="5 years"
    )

    # Initialize session (optional, system instructions are handled by LLMHandler methods internally too)
    await handler.initialize_session("You are an expert technical interviewer.")

    # Generate initial greeting
    initial_greeting = await handler.generate_greeting()
    print(f"AI: {initial_greeting}")

    # Simulate candidate response (e.g., from STT)
    candidate_first_response = "Thank you! I'm excited to be here."
    await handler.process_candidate_response(candidate_first_response)

    # Generate the first actual interview question
    first_question = await handler.generate_interview_question()
    print(f"AI: {first_question}")

    # Simulate candidate answer for the first question
    candidate_answer_1 = "In my previous role, I used Django REST Framework to build APIs for our web application. We focused on proper serialization and authentication to ensure data security."
    await handler.process_candidate_response(candidate_answer_1)

    # Score the first answer
    # You'd need to identify the skill for scoring; for now, let's assume LLM identifies it.
    skill_for_scoring = await handler.llm_identify_skill_from_question(first_question)
    score_result = await handler.score_candidate_answer(first_question, candidate_answer_1, skill_being_assessed=skill_for_scoring)
    print(f"\n--- Scoring for Q1 ---")
    print(json.dumps(score_result, indent=2))
    print(f"----------------------\n")

    # Generate a follow-up question (requires LLMHandler to hold the context of the last question)
    # This is where LLMHandler needs to be smart to pick the right prompt.
    # One way: have a method `generate_next_question_intelligent(last_question, last_answer)`
    # This method internally decides if it's a follow-up or a new question.
    
    # For now, if you want a *specific* follow-up, you'd directly use `prompts.follow_up_prompt`
    # BUT, it needs all the context that LLMHandler has. So the call should be:
    
    # This is an example of calling a specific prompt via get_llm_response_for_task
    # This assumes `first_question` was the *last* question asked.
    last_question_content_from_history = handler.interview_history[-2]['content'] if len(handler.interview_history) >= 2 else ""
    follow_up_prompt_content = prompts.follow_up_prompt.format(
        previous_question=last_question_content_from_history,
        candidate_answer=candidate_answer_1,
        candidate_actual_experience=handler.candidate_actual_experience,
        chosen_interview_level=handler.chosen_interview_level,
        key_skills=", ".join(handler.key_skills)
    )
    system_instruction_followup = "You are an expert interviewer. Generate a concise follow-up question based on the previous answer and context."
    follow_up_q = await handler.get_llm_response_for_task(
        prompt_content=follow_up_prompt_content,
        system_instructions=system_instruction_followup,
        temperature=0.7
    )
    if follow_up_q:
        # Add to history
        handler.interview_history.append({"role": "assistant", "content": follow_up_q})
        print(f"AI (Follow-up): {follow_up_q}")
    else:
        print("Failed to generate follow-up question.")


    # ... continue interview flow ...

    # Generate closing statement
    final_closing = await handler.generate_closing_statement()
    print(f"AI: {final_closing}")

    # End session
    await handler.end_interview_session()


if __name__ == "__main__":
    asyncio.run(main_interview_orchestrator())