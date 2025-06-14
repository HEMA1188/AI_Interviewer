# config.py

import os
import logging
import asyncio
import glob
import shutil
from dotenv import load_dotenv
from datetime import datetime
from distutils.util import strtobool
from logging.handlers import RotatingFileHandler

# === Load Environment Variables ===
# This will load variables from a .env file if it exists in the current directory or parent directories.
if not load_dotenv(override=True):
    print("⚠️ .env file not found. Using system environment variables.")

# === Try to import OpenAI client ===
try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError("🛑 The 'openai' package is missing. Please run `pip install openai`.") from e

# === API Key Validations ===
# OpenAI API Key
OPENAI_API_KEY = os.getenv("API_KEY") # Assuming "API_KEY" in .env maps to OpenAI
if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 20: # Basic length check for API key
    raise ValueError("⚠️ Invalid OPENAI_API_KEY. Please check your .env file or environment variables.")

# D-ID API Key
# Add this line to fetch your D-ID API key from environment variables
D_ID_API_KEY = os.getenv("D_ID_API_KEY")
if not D_ID_API_KEY:
    # It's okay if D-ID key is not set if the feature is optional,
    # but log a warning or raise an error if it's mandatory.
    logging.warning("⚠️ D_ID_API_KEY not found in environment variables. D-ID avatar feature might be disabled or not work.")


# Initialize Async OpenAI Client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# === Debug Mode ===
DEBUG_MODE = bool(strtobool(os.getenv("DEBUG_MODE", "False")))

# === BASE PATHS ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
LOG_DIR = os.path.join(BASE_DIR, "logs")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
REPORTS_DIR = os.path.join(STORAGE_DIR, "reports")
EVALUATIONS_DIR = os.path.join(STORAGE_DIR, "evaluations")
CANDIDATES_DIR = os.path.join(BASE_DIR, "candidates")

LOG_FILE = os.path.join("logs", "interview_ai.log") # This variable seems unused, log_file is defined in setup_logging

# === Create Required Directories ===
def create_directories():
    """Ensures all necessary project directories exist."""
    for path in [TEMP_DIR, LOG_DIR, STORAGE_DIR, REPORTS_DIR, EVALUATIONS_DIR, CANDIDATES_DIR]:
        os.makedirs(path, exist_ok=True)

create_directories()

# === LOGGING CONFIGURATION WITH COLOR ===
def setup_logging():
    """Configures the application-wide logger."""
    log_file = os.path.join(LOG_DIR, "ai_requests.log")
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 3))

    # File handler for persistent logs
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=log_backup_count)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Console handler with optional colorlog
    try:
        import colorlog
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        ))
    except ImportError:
        # Fallback to a standard console handler if colorlog is not installed
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Get the root logger for the application
    logger = logging.getLogger("AI_Interviewer")
    logger.setLevel(log_level)

    # Add handlers only if they haven't been added before to prevent duplicate logs
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logging()
logger.info("✅ Configuration with colorlog loaded successfully.")

# === AI MODEL SETTINGS ===
MODEL_GPT4O = "gpt-4o"
MODEL_GPT4O_MINI = "gpt-4o-mini"
MODEL_GPT3_5_TURBO = "gpt-3.5-turbo"

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", MODEL_GPT4O_MINI)

DEFAULT_PARAMS = {
    "temperature": float(os.getenv("TEMPERATURE", 0.7)),
    "max_tokens": int(os.getenv("MAX_TOKENS", 500)),
    "top_p": float(os.getenv("TOP_P", 1)),
    "frequency_penalty": float(os.getenv("FREQUENCY_PENALTY", 0)),
    "presence_penalty": float(os.getenv("PRESENCE_PENALTY", 0)),
}

# Assertions for valid parameter ranges
assert 0 <= DEFAULT_PARAMS['temperature'] <= 1, "Temperature must be between 0 and 1."
assert 0 <= DEFAULT_PARAMS['top_p'] <= 1, "Top_p must be between 0 and 1."

# === AUDIO PROCESSING SETTINGS ===
AVAILABLE_TTS_PROVIDERS = ["openai"]  # Add other providers if you integrate them
AVAILABLE_STT_ENGINES = ["whisper-1"]  # Add other engines if you integrate them
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

TTS_ENGINE = os.getenv("TTS_ENGINE", "gpt-4o-mini-tts") # This usually refers to the model name if using OpenAI TTS
STT_ENGINE = os.getenv("STT_ENGINE", "whisper-1")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "onyx")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai") # 'openai' suggests using OpenAI's built-in TTS

# Validate audio settings
if TTS_PROVIDER not in AVAILABLE_TTS_PROVIDERS:
    logger.warning(f"⚠️ '{TTS_PROVIDER}' is not a valid TTS provider. Falling back to 'openai'.")
    TTS_PROVIDER = "openai"

if STT_ENGINE not in AVAILABLE_STT_ENGINES:
    logger.warning(f"⚠️ '{STT_ENGINE}' is not a valid STT engine. Falling back to 'whisper-1'.")
    STT_ENGINE = "whisper-1"

if DEFAULT_VOICE not in AVAILABLE_VOICES:
    logger.warning(f"⚠️ '{DEFAULT_VOICE}' is not a valid voice. Falling back to 'onyx'.")
    DEFAULT_VOICE = "onyx"

# === DATABASE CONFIGURATION (PostgreSQL) ===
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "interview_db")
DB_USER = os.getenv("DB_USER", "interview_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "interview_password")

# Connection string for asyncpg (optional, can be constructed in code too)
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- INTERVIEW SETTINGS ---
# New: Interview slot duration (e.g., "30_min", "60_min")
INTERVIEW_SLOT_DURATION_STR = os.getenv("INTERVIEW_SLOT_DURATION", "30_min").lower()

if INTERVIEW_SLOT_DURATION_STR == "30_min":
    TOTAL_INTERVIEW_DURATION_SECONDS = 30 * 60  # 30 minutes in seconds
    QUESTION_LIMIT = int(os.getenv("QUESTION_LIMIT", 7)) # Adjust based on 30 min duration
    TIME_PER_QUESTION = int(os.getenv("TIME_PER_QUESTION", 120)) # 2 minutes (120 seconds) per question
elif INTERVIEW_SLOT_DURATION_STR == "60_min":
    TOTAL_INTERVIEW_DURATION_SECONDS = 60 * 60  # 60 minutes in seconds
    QUESTION_LIMIT = int(os.getenv("QUESTION_LIMIT", 15)) # Adjust based on 60 min duration
    TIME_PER_QUESTION = int(os.getenv("TIME_PER_QUESTION", 180)) # 3 minutes (180 seconds) per question
else:
    logger.warning(f"⚠️ Invalid INTERVIEW_SLOT_DURATION: '{INTERVIEW_SLOT_DURATION_STR}'. Defaulting to '30_min'.")
    TOTAL_INTERVIEW_DURATION_SECONDS = 30 * 60
    QUESTION_LIMIT = int(os.getenv("QUESTION_LIMIT", 7))
    TIME_PER_QUESTION = int(os.getenv("TIME_PER_QUESTION", 120)) # Default to 2 minutes

# Export the final calculated duration
TOTAL_INTERVIEW_DURATION = TOTAL_INTERVIEW_DURATION_SECONDS

# Behavior flags and fallbacks
SILENCE_TIMEOUT = int(os.getenv("SILENCE_TIMEOUT", 15)) # Seconds of silence before fallback
MAX_CONSECUTIVE_SILENCES = int(os.getenv("MAX_CONSECUTIVE_SILENCES", 2)) # How many times to tolerate silence

ALLOW_SILENCE_HANDLING = bool(strtobool(os.getenv("ALLOW_SILENCE_HANDLING", "True")))
ALLOW_REPHRASE_ON_IRRELEVANT = bool(strtobool(os.getenv("ALLOW_REPHRASE_ON_IRRELEVANT", "True")))

if ALLOW_SILENCE_HANDLING:
    logger.debug("🟢 Silence detection is enabled.")
if ALLOW_REPHRASE_ON_IRRELEVANT:
    logger.debug("🟢 Irrelevant response handling is enabled.")

# === FALLBACK RESPONSES ===
FALLBACK_RESPONSE = os.getenv("FALLBACK_RESPONSE", "It seems you didn’t respond. Would you like me to repeat the question?")
IRRELEVANT_RESPONSE = os.getenv("IRRELEVANT_RESPONSE", "Hmm, that doesn’t seem to match the question. Let me rephrase or try another.")

# === UTILITY FUNCTIONS ===
def timestamp():
    """Generates a formatted current timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def test_openai():
    """Tests the OpenAI API connection."""
    try:
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "system", "content": "Say hello!"}],
            **DEFAULT_PARAMS
        )
        logger.info(f"Test Response from OpenAI: {response.choices[0].message.content}")
    except Exception as e:
        logger.error(f"❌ Error in OpenAI API test request: {e}")

def cleanup_logs():
    """Cleans up old log files based on LOG_BACKUP_COUNT."""
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "ai_requests.log*")), reverse=True)
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 3))
    # Keep the latest current log file and the specified number of backups
    if len(log_files) > log_backup_count + 1:
        for old_log in log_files[log_backup_count + 1:]:
            try:
                # Check if it's a directory (e.g., from old backup strategies) or a file
                if os.path.isdir(old_log):
                    shutil.rmtree(old_log)
                else:
                    os.remove(old_log)
                logger.info(f"🗑️ Deleted old log file: {old_log}")
            except Exception as e:
                logger.error(f"❌ Error deleting log file {old_log}: {e}")

cleanup_logs() # Execute log cleanup on startup

# === GLOBAL CONFIG EXPORT ===
# A dictionary to easily access all major configuration settings
CONFIG = {
    "model": DEFAULT_MODEL,
    "params": DEFAULT_PARAMS,
    "voice": DEFAULT_VOICE,
    "question_limit": QUESTION_LIMIT,
    "interview_duration": TOTAL_INTERVIEW_DURATION,
    "time_per_question": TIME_PER_QUESTION,
    "audio": {
        "tts": TTS_ENGINE,
        "stt": STT_ENGINE,
        "provider": TTS_PROVIDER
    },
    "behavior_flags": {
        "silence_handling": ALLOW_SILENCE_HANDLING,
        "rephrase_irrelevant": ALLOW_REPHRASE_ON_IRRELEVANT
    },
    "fallbacks": {
        "silence": FALLBACK_RESPONSE,
        "irrelevant": IRRELEVANT_RESPONSE
    },
    "paths": {
        "temp": TEMP_DIR,
        "logs": LOG_DIR,
        "reports": REPORTS_DIR,
        "evaluations": EVALUATIONS_DIR,
        "storage": STORAGE_DIR
    }
}

# === EVENT LOOP HANDLING (for module testing) ===
# This block ensures test_openai() runs only when config.py is executed directly.
if __name__ == "__main__":
    try:
        # Attempt to get an existing running loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no loop is running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the OpenAI test
    loop.run_until_complete(test_openai())