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
if not load_dotenv(override=True):
    print("⚠️ .env file not found. Using system environment variables.")

# === Try to import OpenAI client ===
try:
    from openai import AsyncOpenAI
except ImportError as e:
    raise ImportError("🛑 The 'openai' package is missing. Please run `pip install openai`.") from e

# === OpenAI API Key Validation ===
OPENAI_API_KEY = os.getenv("API_KEY")
if not OPENAI_API_KEY or len(OPENAI_API_KEY) < 20:
    raise ValueError("⚠️ Invalid OPENAI_API_KEY. Please check your .env file or environment variables.")

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

LOG_FILE = os.path.join("logs", "interview_ai.log")

# === Create Required Directories ===
def create_directories():
    for path in [TEMP_DIR, LOG_DIR, STORAGE_DIR, REPORTS_DIR, EVALUATIONS_DIR, CANDIDATES_DIR]:
        os.makedirs(path, exist_ok=True)

create_directories()

# === LOGGING CONFIGURATION WITH COLOR ===
def setup_logging():
    log_file = os.path.join(LOG_DIR, "ai_requests.log")
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 3))

    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=log_backup_count)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

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
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger("AI_Interviewer")
    logger.setLevel(log_level)

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

assert 0 <= DEFAULT_PARAMS['temperature'] <= 1
assert 0 <= DEFAULT_PARAMS['top_p'] <= 1

# === AUDIO PROCESSING SETTINGS ===
AVAILABLE_TTS_PROVIDERS = ["openai"]  # Add other providers if you integrate them
AVAILABLE_STT_ENGINES = ["whisper-1"]  # Add other engines if you integrate them
AVAILABLE_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

TTS_ENGINE = os.getenv("TTS_ENGINE", "gpt-4o-mini-tts")
STT_ENGINE = os.getenv("STT_ENGINE", "whisper-1")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "onyx")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai")

if TTS_PROVIDER not in AVAILABLE_TTS_PROVIDERS:
    logger.warning(f"⚠️ '{TTS_PROVIDER}' is not a valid TTS provider. Falling back to 'openai'.")
    TTS_PROVIDER = "openai"

if STT_ENGINE not in AVAILABLE_STT_ENGINES:
    logger.warning(f"⚠️ '{STT_ENGINE}' is not a valid STT engine. Falling back to 'whisper-1'.")
    STT_ENGINE = "whisper-1"

if DEFAULT_VOICE not in AVAILABLE_VOICES:
    logger.warning(f"⚠️ '{DEFAULT_VOICE}' is not a valid voice. Falling back to 'onyx'.")
    DEFAULT_VOICE = "onyx"

# PostgreSQL Database Configuration
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
    QUESTION_LIMIT = int(os.getenv("QUESTION_LIMIT", 5)) # Adjust based on 30 min
    TIME_PER_QUESTION = int(os.getenv("TIME_PER_QUESTION", 30)) # 2 minutes per question
elif INTERVIEW_SLOT_DURATION_STR == "60_min":
    TOTAL_INTERVIEW_DURATION_SECONDS = 60 * 60  # 60 minutes in seconds
    QUESTION_LIMIT = int(os.getenv("QUESTION_LIMIT", 15)) # Adjust based on 60 min
    TIME_PER_QUESTION = int(os.getenv("TIME_PER_QUESTION", 60)) # 3 minutes per question
else:
    logger.warning(f"⚠️ Invalid INTERVIEW_SLOT_DURATION: '{INTERVIEW_SLOT_DURATION_STR}'. Defaulting to '30_min'.")
    TOTAL_INTERVIEW_DURATION_SECONDS = 30 * 60
    QUESTION_LIMIT = int(os.getenv("QUESTION_LIMIT", 7))
    TIME_PER_QUESTION = int(os.getenv("TIME_PER_QUESTION", 30))


# Export the final calculated duration
TOTAL_INTERVIEW_DURATION = TOTAL_INTERVIEW_DURATION_SECONDS

SILENCE_TIMEOUT = int(os.getenv("SILENCE_TIMEOUT", 15))
MAX_CONSECUTIVE_SILENCES = int(os.getenv("MAX_CONSECUTIVE_SILENCES", 2))

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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def test_openai():
    try:
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "system", "content": "Say hello!"}],
            **DEFAULT_PARAMS
        )
        logger.info(f"Test Response: {response.choices[0].message.content}")
    except Exception as e:
        logger.error(f"❌ Error in OpenAI request: {e}")

def cleanup_logs():
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "ai_requests.log*")), reverse=True)
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 3))
    if len(log_files) > log_backup_count + 1:  # Keep the latest and backup count
        for old_log in log_files[log_backup_count + 1:]:
            try:
                shutil.rmtree(old_log) if os.path.isdir(old_log) else os.remove(old_log)
                logger.info(f"🗑️ Deleted old log file: {old_log}")
            except Exception as e:
                logger.error(f"❌ Error deleting log {old_log}: {e}")

cleanup_logs()

# === GLOBAL CONFIG EXPORT ===
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

# === EVENT LOOP HANDLING ===
if __name__ == "__main__":
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(test_openai())