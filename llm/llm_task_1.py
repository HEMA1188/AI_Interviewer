import asyncio
import json
import logging
import os
import sys
from typing import List, Optional, Dict, AsyncGenerator
# Assuming AsyncOpenAI is correctly imported from your environment
from openai import AsyncOpenAI # Make sure openai library is installed (pip install openai)
import traceback

# === Setup Project Path (adjust if needed) ===
# This ensures that imports like 'utils.config' and 'resources.prompts_3' work correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === Imports after sys.path is updated ===
# These imports rely on PROJECT_ROOT being in sys.path
try:
    from utils.config import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_PARAMS, client
    from resources.prompts_3 import prompts
except ImportError as e:
    logging.error(f"Failed to import from utils.config or resources.prompts_3. "
                  f"Please ensure these files exist and are correctly configured. Error: {e}")
    # Exit or handle gracefully, as core dependencies are missing
    sys.exit(1)

# === Logging Setup ===
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Added format for better logs

# --- Helper Functions for Skill Prioritization (now based on Resume) ---
def determine_skill_priority_from_resume(resume_text: str, all_potential_skills: List[str]) -> List[str]:
    """
    Dynamically determines the priority of skills based on their presence and frequency in the Resume.
    """
    skill_priority = {}
    resume_lower = resume_text.lower()

    for skill in all_potential_skills:
        skill_lower = skill.lower()
        if skill_lower in resume_lower:
            # Simple heuristic: count occurrences to infer importance
            count = resume_lower.count(skill_lower)
            skill_priority[skill] = count
        else:
            skill_priority[skill] = 0

    # Sort skills by priority (descending)
    sorted_skills = sorted(skill_priority.items(), key=lambda item: item[1], reverse=True)
    return [skill for skill, priority in sorted_skills if priority > 0] # Only return skills found in resume

def get_weights(skill_being_assessed: str, prioritized_skills: List[str]) -> tuple[int, int]:
    """
    Dynamically assigns weights based on the specific skill being assessed
    and its priority within the candidate's resume.
    """
    relevance_weight = 1
    depth_weight = 1 # Default weights

    if not prioritized_skills:
        return relevance_weight, depth_weight

    try:
        # Find the best match for the skill being assessed from the prioritized list
        matched_skill_index = -1
        for i, p_skill in enumerate(prioritized_skills):
            # Check for exact match or strong substring match
            if skill_being_assessed.lower() == p_skill.lower() or \
               (skill_being_assessed.lower() in p_skill.lower() and len(skill_being_assessed) > 3) or \
               (p_skill.lower() in skill_being_assessed.lower() and len(p_skill) > 3):
                matched_skill_index = i
                break

        if matched_skill_index != -1:
            # Assign weights based on priority tiers from resume
            if matched_skill_index < 2:  # Top 2 prioritized skills from resume
                relevance_weight = 3
                depth_weight = 2
            elif matched_skill_index < 4:  # Next 2 prioritized skills from resume
                relevance_weight = 2
                depth_weight = 1
            else:  # Lower prioritized skills from resume
                relevance_weight = 1
                depth_weight = 1
        else:
            logger.debug(f"Skill '{skill_being_assessed}' not found directly in prioritized skills from resume. Using default weights.")
    except Exception as e:
        logger.error(f"Error determining weights for skill '{skill_being_assessed}': {e}")
        # Default weights already set

    return relevance_weight, depth_weight


class LLMHandler:
    """Handles interactions with the OpenAI Chat Completions API for the interview flow."""

    def __init__(self, client: AsyncOpenAI, model: str = DEFAULT_MODEL, temperature: float = 0.7):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.interview_history: List[Dict[str, str]] = [] # Stores history as messages (role, content)
        self.candidate_resume: Optional[str] = None # Stores the candidate's resume
        self.chosen_interview_level: Optional[str] = None # This is the chosen interview level (Basic/Intermediate/Advanced)
        self.candidate_actual_experience: Optional[str] = None # e.g., "fresher", "1 year", "5 years"
        self.key_skills: Optional[List[str]] = None # Stores extracted key skills from resume
        self.candidate_name: Optional[str] = None # Storing candidate name for greeting/closing

    async def initialize_session(self, initial_instructions: str):
        """
        Initializes the interview session by setting up the system prompt.
        """
        self.interview_history.append({"role": "system", "content": initial_instructions})
        logger.info("Interview session initialized with system instructions.")
        return True

    async def set_interview_context(self, candidate_name: str, resume: str, chosen_interview_level: str, candidate_actual_experience: str):
        """
        Sets the overall interview context including candidate info, resume, actual experience,
        and the chosen interview difficulty level. It also extracts/prioritizes skills based on the resume.
        This should be called once at the start of an interview session.
        """
        self.candidate_name = candidate_name # Store candidate name
        self.candidate_resume = resume
        # Ensure consistent casing (Basic, Intermediate, Advanced) for the chosen level
        self.chosen_interview_level = chosen_interview_level.capitalize()
        # Store the candidate's actual experience directly
        self.candidate_actual_experience = candidate_actual_experience

        logger.info(f"Setting interview context. Candidate actual experience: {self.candidate_actual_experience}, Chosen interview level: {self.chosen_interview_level}")

        # Step 1: Use LLM to extract key skills from the resume
        system_instruction_for_skill_extraction = "You are an expert at extracting key technical and soft skills from resumes. Respond with a JSON array of skill names as specified. ONLY provide the JSON."
        extracted_skills_json_str = await self.get_llm_response_for_task(
            prompt_content=prompts.EXTRACT_SKILLS_PROMPT.format(resume=resume),
            system_instructions=system_instruction_for_skill_extraction,
            temperature=0.0 # Keep temperature low for factual extraction
        )

        try:
            # Attempt to parse the JSON, handling potential markdown fences
            if extracted_skills_json_str.strip().startswith("```json"):
                extracted_skills_json_str = extracted_skills_json_str.strip()[len("```json"):].strip()
                if extracted_skills_json_str.strip().endswith("```"):
                    extracted_skills_json_str = extracted_skills_json_str.strip()[:-len("```")].strip()

            extracted_skills_data = json.loads(extracted_skills_json_str)
            self.key_skills = extracted_skills_data.get("skills", [])
            if not isinstance(self.key_skills, list): # Ensure it's a list
                self.key_skills = []
                logger.warning(f"Extracted skills 'skills' field was not a list: {extracted_skills_json_str}")

        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError extracting skills from resume: {e}. Raw LLM response: {extracted_skills_json_str}", exc_info=True)
            self.key_skills = [] # Fallback to empty list
        except Exception as e:
            logger.error(f"Unexpected error extracting skills: {e}", exc_info=True)
            self.key_skills = []

        logger.info(f"Interview context set for {candidate_name}. Actual experience: {self.candidate_actual_experience}, Chosen Level: {self.chosen_interview_level}. Extracted key skills: {self.key_skills}")

    async def generate_greeting(self) -> str:
        """Generates a warm greeting for the candidate."""
        if not all([self.candidate_name, self.candidate_resume, self.candidate_actual_experience, self.chosen_interview_level]):
            logger.error("Cannot generate greeting: Candidate name, resume, actual experience, or chosen interview level not set.")
            return "Hello! Welcome to the interview."

        greeting_prompt_content = prompts.greet_warmup_prompt.format(
            candidate_name=self.candidate_name,
            resume=self.candidate_resume,
            candidate_actual_experience=self.candidate_actual_experience,
            chosen_interview_level=self.chosen_interview_level
        )
        system_instructions = "You are an expert interviewer. Generate a warm and professional greeting."

        greeting = await self.get_llm_response_for_task(
            prompt_content=greeting_prompt_content,
            system_instructions=system_instructions,
            temperature=0.7
        )
        # Add the greeting to history as assistant role for conversation flow
        self.interview_history.append({"role": "assistant", "content": greeting})
        return greeting


    async def process_candidate_response(self, response_text: str):
        """Adds the candidate's response to the interview history."""
        self.interview_history.append({"role": "user", "content": response_text})
        logger.info(f"Candidate response added to history: {response_text[:50]}...")

    async def get_next_assistant_response(self, additional_instructions: Optional[str] = None) -> Optional[str]:
        """
        Sends the current interview history to the LLM and retrieves its response.
        """
        messages_to_send = list(self.interview_history) # Copy to avoid modifying original

        if additional_instructions:
            # Append additional instructions as a temporary system message for this turn
            # This doesn't permanently alter the initial system message but adds context for the current turn
            messages_to_send.append({"role": "system", "content": additional_instructions})
            logger.debug(f"Additional instructions for this turn: {additional_instructions}")

        try:
            logger.info(f"Calling OpenAI Chat Completions with {len(messages_to_send)} messages...")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send,
                temperature=self.temperature,
                # Filter DEFAULT_PARAMS to include only relevant ones for chat completions
                **{k: v for k, v in DEFAULT_PARAMS.items() if k in ["max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}
            )
            assistant_response = response.choices[0].message.content.strip()
            self.interview_history.append({"role": "assistant", "content": assistant_response})
            logger.info(f"Assistant response received: {assistant_response[:50]}...")
            return assistant_response
        except Exception as e:
            logger.error(f"Error during LLM response retrieval in get_next_assistant_response: {e}", exc_info=True)
            return None

    async def llm_identify_skill_from_question(self, question: str) -> str:
        """
        Uses the LLM to identify the primary skill being assessed by a question
        in the context of the candidate's resume.
        """
        if not self.candidate_resume:
            logger.error("Candidate resume not set for skill identification.")
            return "General Competency" # Fallback

        prompt_content = prompts.IDENTIFY_SKILL_PROMPT.format(
            resume=self.candidate_resume,
            question=question
        )

        try:
            messages = [
                {"role": "system", "content": "You are an expert at identifying the core skill or competency being assessed by an interview question, given a candidate's resume. Respond with only the skill name. Do not include any other text or punctuation."},
                {"role": "user", "content": prompt_content}
            ]
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0 # Low temperature for factual extraction
            )
            skill_name = response.choices[0].message.content.strip()
            # Clean up potential unwanted characters if LLM adds them
            return skill_name.replace('.', '').replace('"', '').strip()
        except Exception as e:
            logger.error(f"Error identifying skill for question '{question}': {e}", exc_info=True)
            return "General Competency" # Fallback on error

    async def generate_closing_statement(self) -> str:
        """Generates the closing statement for the interview."""
        closing_prompt_content = prompts.closing_prompt
        system_instructions = "You are an expert interviewer. Generate a brief, professional closing statement and ask if the candidate has any questions."

        closing_statement = await self.get_llm_response_for_task(
            prompt_content=closing_prompt_content,
            system_instructions=system_instructions,
            temperature=0.7
        )
        self.interview_history.append({"role": "assistant", "content": closing_statement})
        return closing_statement

    async def end_interview_session(self):
        """Clears the interview history, effectively ending the session."""
        self.interview_history = []
        self.candidate_resume = None
        self.chosen_interview_level = None # Clear chosen level
        self.candidate_actual_experience = None # Clear actual experience
        self.key_skills = None # Clear extracted skills
        self.candidate_name = None # Clear candidate name
        logger.info("Interview session ended and history cleared.")

    async def get_llm_response_for_task(self, prompt_content: str, system_instructions: str,
                                         model: Optional[str] = None, temperature: Optional[float] = None) -> str:
        """
        General purpose function to interact with the LLM for specific, non-conversational tasks.
        Allows overriding model and temperature for specific tasks.
        """
        model = model or self.model # Use self.model as default if not overridden
        temperature = temperature if temperature is not None else self.temperature # Use self.temperature as default

        response_content = None

        try:
            messages = [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt_content}
            ]
            logger.debug(f"Calling LLM for task with instructions: {system_instructions[:50]}...")
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                # Filter DEFAULT_PARAMS to include only relevant ones for chat completions
                **{k: v for k, v in DEFAULT_PARAMS.items() if k in ["max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}
            )
            response_content = response.choices[0].message.content.strip()
            logger.debug(f"Task LLM response: {response_content[:50]}...")

        except Exception as e:
            logger.error(f"Error during LLM task execution: {e}", exc_info=True)
            response_content = f"Error processing task: {e}"

        return response_content if response_content is not None else ""

    async def generate_interview_question(self, skill_to_assess: Optional[str] = None) -> Optional[str]:
        """
        Generates a single interview question based on the candidate's resume, key skills,
        actual experience, and the chosen interview difficulty level.
        Args:
            skill_to_assess (Optional[str]): A specific skill to focus the question on.
                                              If None, the LLM will pick a relevant skill.
        Returns:
            Optional[str]: The generated question, or None if context is missing or generation fails.
        """
        # Ensure all necessary context is set
        if not all([self.candidate_resume, self.key_skills, self.chosen_interview_level, self.candidate_actual_experience]):
            logger.error("Cannot generate question: Resume, key skills, chosen interview level, or actual experience not set. Call set_interview_context first.")
            return None

        # Prepare instruction for skill_to_assess
        skill_instruction = ""
        if skill_to_assess:
            skill_instruction = f"Specifically, generate a question that assesses the candidate's proficiency in: {skill_to_assess}. Focus on this skill."

        # Format key skills for the prompt
        key_skills_formatted = ", ".join(self.key_skills) if self.key_skills else "general professional competencies"

        prompt_content = prompts.GENERATE_QUESTION_PROMPT.format(
            resume=self.candidate_resume,
            candidate_actual_experience=self.candidate_actual_experience,
            key_skills=key_skills_formatted,
            chosen_interview_level=self.chosen_interview_level,
            skill_to_assess_instruction=skill_instruction
        )

        system_instructions = "You are a specialized AI assistant that generates concise, single interview questions based on provided context. The question should be challenging and directly relevant."

        try:
            logger.info(f"Generating question for chosen level '{self.chosen_interview_level}', actual experience '{self.candidate_actual_experience}', focusing on '{skill_to_assess}' if provided...")
            generated_question = await self.get_llm_response_for_task(
                prompt_content=prompt_content,
                system_instructions=system_instructions,
                temperature=0.7 # A bit higher temperature for more diverse questions
            )
            # Add the generated question to history as assistant role
            self.interview_history.append({"role": "assistant", "content": generated_question})
            logger.info(f"Generated question: {generated_question[:100]}...")
            return generated_question
        except Exception as e:
            logger.error(f"Failed to generate interview question: {e}", exc_info=True)
            return None

    async def score_candidate_answer(self, question: str, candidate_answer: str,
                                       skill_being_assessed: Optional[str] = None) -> Dict:
        """
        Scores the candidate's answer using a dedicated LLM call, based on resume, actual experience,
        and chosen interview difficulty.
        Args:
            question (str): The specific question asked.
            candidate_answer (str): The candidate's response.
            skill_being_assessed (Optional[str]): The specific skill this question is primarily assessing.
        Returns:
            Dict: The parsed JSON scoring result, or an empty dictionary on failure.
        """
        if not all([self.candidate_resume, self.key_skills, self.chosen_interview_level, self.candidate_actual_experience]):
            logger.error("Candidate resume, key skills, chosen interview level, or actual experience not set. Call set_interview_context first.")
            return {}

        # Determine weights based on the specific skill being assessed
        # Note: get_weights might not be strictly necessary if prompt relies solely on LLM to interpret skill importance
        # But it's kept for now as per your original structure.
        # Ensure self.key_skills is a list when passed
        prioritized_skills_list = self.key_skills if self.key_skills is not None else []
        relevance_weight, depth_weight = get_weights(skill_being_assessed or question, prioritized_skills_list)


        # Format key skills for the prompt
        key_skills_formatted = ", ".join(self.key_skills) if self.key_skills else "N/A"

        scoring_prompt_content = prompts.SCORING_CRITERIA_PROMPT.format(
            resume=self.candidate_resume,
            candidate_actual_experience=self.candidate_actual_experience,
            question=question,
            candidate_answer=candidate_answer,
            chosen_interview_level=self.chosen_interview_level,
            key_skills=key_skills_formatted # Pass the formatted key skills
        )
        scoring_system_instructions = "You are an expert interviewer and evaluator. Score candidate answers based on the provided resume, question, and the candidate's response. Output the score and rationale in a JSON format as specified in the prompt. Ensure the output is valid JSON and only contains the JSON."

        for attempt in range(3):
            logger.info(f"Attempt {attempt + 1} to score candidate answer...")
            response_json_str = await self.get_llm_response_for_task(
                prompt_content=scoring_prompt_content,
                system_instructions=scoring_system_instructions,
                model=self.model,
                temperature=0.7 # Using self.temperature by default, or you can override here
            )
            try:
                # Clean up the JSON string: remove any leading/trailing text, especially common markdown
                if response_json_str.strip().startswith("```json"):
                    response_json_str = response_json_str.strip()[len("```json"):].strip()
                    if response_json_str.strip().endswith("```"):
                        response_json_str = response_json_str.strip()[:-len("```")].strip()

                response_data = json.loads(response_json_str)
                return response_data
            except json.JSONDecodeError as e:
                logger.error(
                    f"JSONDecodeError (Attempt {attempt + 1}): {e}. Failed to decode JSON from LLM response. Raw LLM Response:\n{response_json_str}")
                if attempt < 2:
                    await asyncio.sleep(1) # Wait before retrying
                else:
                    logger.error("Max retries reached for JSON decoding in score_candidate_answer. Returning empty dict.")
                    return {}
            except Exception as e:
                logger.error(f"Error processing LLM response in score_candidate_answer: {e}. Raw response: {response_json_str}", exc_info=True)
                return {}
        return {} # Should not be reached but for safety

    async def generate_final_report(self) -> str:
        """Generates a final report using a dedicated LLM call, based on the resume, actual experience, and chosen interview level."""
        if not all([self.candidate_resume, self.interview_history, self.key_skills, self.chosen_interview_level, self.candidate_actual_experience]):
            logger.error("Cannot generate final report: Resume, history, key skills, chosen interview level, or actual experience missing.")
            return "Error: Interview context not fully set."

        key_skills_str = ", ".join(self.key_skills) if self.key_skills else "N/A" # Use all extracted key skills
        
        # Format interview history for the prompt
        formatted_history = []
        # Exclude greeting, closing, and internal system messages from history for report
        for msg in self.interview_history:
            if msg['role'] == 'system':
                continue
            # Optionally, you might filter out the initial greeting if it's too generic
            # For now, let's include all non-system user/assistant messages
            formatted_history.append(f"{msg['role'].capitalize()}: {msg['content']}")

        report_prompt_content = prompts.FINAL_REPORT_PROMPT.format(
            history="\n".join(formatted_history),
            resume=self.candidate_resume,
            candidate_actual_experience=self.candidate_actual_experience,
            chosen_interview_level=self.chosen_interview_level,
            key_skills=key_skills_str
        )
        report_system_instructions = "You are an expert HR professional and interviewer. Generate a comprehensive final interview report based on the provided interview history and candidate's resume. Structure the report as a professional dashboard, covering strengths, weaknesses, skill assessment, and overall recommendation. Provide a clear, well-structured report suitable for HR review."

        return await self.get_llm_response_for_task(
            prompt_content=report_prompt_content,
            system_instructions=report_system_instructions,
            model=self.model,
            temperature=0.7
        )

    async def get_improvement_suggestions(self, question: str, candidate_answer: str) -> str:
        """Provides suggestions for improving the candidate's answer using a dedicated LLM call."""
        if not all([self.chosen_interview_level, self.candidate_resume, self.candidate_actual_experience]):
            logger.error("Candidate chosen interview level, resume, or actual experience not set for improvement suggestions.")
            return "Error: Context not fully set."

        improvement_prompt_content = prompts.IMPROVEMENT_PROMPT.format(
            question=question,
            candidate_answer=candidate_answer,
            chosen_interview_level=self.chosen_interview_level,
            resume=self.candidate_resume, # Add resume to the prompt for context
            candidate_actual_experience=self.candidate_actual_experience # Pass candidate's actual experience
        )
        improvement_system_instructions = "You are an expert interview coach. Provide specific, actionable, and constructive ways a candidate could improve their interview answer to the given question. Focus on clarity, completeness, relevance, and depth. Provide only the suggestions, no conversational filler."

        return await self.get_llm_response_for_task(
            prompt_content=improvement_prompt_content,
            system_instructions=improvement_system_instructions,
            model=self.model,
            temperature=0.7
        )

# === Global helper functions (now directly use Chat Completions or the handler) ===
# These are still provided for backward compatibility with existing calls,
# but it's recommended to use the LLMHandler methods directly.

async def llm_generate_final_report_global(handler: LLMHandler) -> str:
    """Generates the final interview report using the Chat Completions API via handler."""
    return await handler.generate_final_report()

async def llm_get_improvement_suggestions_global(handler: LLMHandler, question: str,
                                                 candidate_answer: str) -> str:
    """Gets improvement suggestions using the Chat Completions API via handler."""
    return await handler.get_improvement_suggestions(question, candidate_answer)

# === Call OpenAI Stream (Standard Chat Completions API) ===
async def call_openai_stream(prompt: str, temperature: Optional[float] = None, model: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
    model = model or DEFAULT_MODEL
    temperature = temperature if temperature is not None else DEFAULT_PARAMS.get("temperature")
    params = {**DEFAULT_PARAMS, **kwargs}
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
            **{k: v for k, v in params.items() if k in ["max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.exception("LLM streaming call failed")
        yield f"❌ Error: {e}\n{traceback.format_exc()}"

# === Call OpenAI Standard (Standard Chat Completions API) ===
async def call_openai(prompt: str, temperature: Optional[float] = None, model: Optional[str] = None, **kwargs) -> str:
    model = model or DEFAULT_MODEL
    temperature = temperature if temperature is not None else DEFAULT_PARAMS.get("temperature")
    params = {**DEFAULT_PARAMS, **kwargs}
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **{k: v for k, v in params.items() if k in ["max_tokens", "top_p", "frequency_penalty", "presence_penalty"]}
        )
        output = response.choices[0].message.content.strip()
        return output
    except Exception as e:
        logger.exception("LLM call failed")
        return f"❌ LLM call failed: {e}\n{traceback.format_exc()}"