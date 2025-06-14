import os
import sys
import asyncio
import logging
import tempfile
import wave
from typing import List, Optional

import numpy as np
import sounddevice as sd
import noisereduce as nr
from pydub import AudioSegment
from pydub.silence import detect_silence
from openai import AsyncOpenAI
import io

# === Windows-specific fix ===
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# === Project Setup ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# === Configuration ===
try:
    from utils.config import OPENAI_API_KEY, STT_ENGINE # Import STT_ENGINE here
except ImportError:
    logging.error("Failed to import OPENAI_API_KEY or STT_ENGINE from utils.config. Please ensure config.py exists and is correctly configured.")
    sys.exit(1)


# === Logging ===
# Get logger instance; its configuration should be handled by the main application
logger = logging.getLogger(__name__)

# === API Key Check ===
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Please set it in your environment variables or config file.")

# === Directory Setup ===
AUDIO_DIR = os.path.abspath(os.path.join("storage", "audio"))
os.makedirs(AUDIO_DIR, exist_ok=True)

# === OpenAI Client ===
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# === STT and Audio Processing Class ===
class STT:
    """Handles audio recording, noise reduction, silence detection, and transcription."""
    def __init__(self, model: str = STT_ENGINE):
        self.client = client
        self.model = model

    @staticmethod
    def reduce_noise(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Reduces noise from the input audio data."""
        try:
            # Noise reduction expects float64, so ensure correct dtype
            return nr.reduce_noise(y=audio_data.astype(np.float64), sr=sample_rate)
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return audio_data

    @staticmethod
    def detect_silence_in_audio(audio_path: str, silence_thresh: int = -40, min_silence_len: int = 1000) -> List[List[int]]:
        """
        Detects silence segments in an audio file.
        Args:
            audio_path: Path to the WAV audio file.
            silence_thresh: Silence threshold in dBFS (decibels relative to full scale).
                            Lower values mean more sensitive to quiet sounds.
            min_silence_len: Minimum length of silence in milliseconds to be considered silent.
        Returns:
            A list of lists, where each inner list contains [start_ms, end_ms] of a silence segment.
        """
        try:
            sound = AudioSegment.from_wav(audio_path)
            return detect_silence(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        except Exception as e:
            logger.error(f"Silence detection failed for {audio_path}: {e}")
            return []

    @staticmethod
    def generate_audio_path() -> str:
        """Generates a unique temporary audio file path."""
        # Use a more robust way to generate unique filenames if tempfile._get_candidate_names()
        # is causing issues or if you want more control.
        # For example: f"recording_{uuid.uuid4().hex}.wav" if uuid is imported.
        return os.path.join(AUDIO_DIR, f"recording_{next(tempfile._get_candidate_names())}.wav")

    @staticmethod
    def record_audio(duration: int = 10, samplerate: int = 16000) -> str:
        """Records audio for a specified duration and saves it to a WAV file (synchronous)."""
        try:
            if duration <= 0 or samplerate <= 0:
                raise ValueError("Duration and samplerate must be positive integers.")
            
            # Check for available input devices
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if not input_devices:
                logger.error("No audio input devices found. Please check your microphone connection.")
                return ""

            logger.info(f"Recording audio for {duration} seconds (Samplerate: {samplerate} Hz)...")
            audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait() # Wait for recording to finish

            # Apply noise reduction
            audio_data = STT.reduce_noise(audio_data.flatten(), samplerate)
            
            # Normalize audio to prevent clipping when converting to int16
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = (audio_data / max_val) * 32767
            else: # Handle case of completely silent audio
                audio_data = np.zeros_like(audio_data)

            audio_path = STT.generate_audio_path()
            with wave.open(audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 2 bytes for int16
                wf.setframerate(samplerate)
                wf.writeframes(audio_data.astype(np.int16).tobytes()) # Convert to int16 bytes

            logger.info(f"Audio recorded at {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            # Log available devices to help debug
            try:
                logger.error(f"Available input devices: {sd.query_devices(kind='input')}")
            except Exception as e_dev:
                logger.error(f"Could not query devices: {e_dev}")
            return ""

    @staticmethod
    async def get_audio_duration(audio_path: str) -> float:
        """Asynchronously gets the duration of an audio file in seconds."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, STT._get_audio_duration_sync, audio_path)
        except Exception as e:
            logger.error(f"Error getting audio duration for {audio_path}: {e}")
            return 0.0

    @staticmethod
    def _get_audio_duration_sync(audio_path: str) -> float:
        """Synchronously gets the duration of an audio file in seconds."""
        try:
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found for duration check: {audio_path}")
                return 0.0
            with wave.open(audio_path, 'rb') as wf:
                frame_rate = wf.getframerate()
                num_frames = wf.getnframes()
                duration = num_frames / frame_rate
                return duration
        except Exception as e:
            logger.error(f"Error reading audio file {audio_path} for duration: {e}")
            return 0.0

    async def transcribe_audio(self, audio_path: str, language: str = None) -> dict:
        """
        Converts audio from a file to text using Whisper (asynchronous).
        Returns a dictionary with 'text' and 'status'.
        """
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"Audio file not found for transcription: {audio_path}")
            return {"text": None, "status": "error", "message": "Audio file not found."}
        
        try:
            with open(audio_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="text",
                    language=language
                )
            text = response.strip()
            logger.info(f"Raw transcription result: '{text}'") # Log raw text for debugging

            if not text:
                silence_segments = self.detect_silence_in_audio(audio_path)
                audio_duration_ms = (await self.get_audio_duration(audio_path)) * 1000 # Convert to milliseconds
                
                total_silence_duration = sum([end - start for start, end in silence_segments])
                
                # If total silence covers a significant portion (e.g., > 80%) of the recording
                if audio_duration_ms > 0 and total_silence_duration >= 0.80 * audio_duration_ms:
                    return {"text": "", "status": "silent", "message": "You were silent, or your voice was not audible. Please check your microphone and try speaking clearly."}
                else:
                    return {"text": "", "status": "no_speech", "message": "No clear speech detected. Please try again, ensuring your voice is loud and clear."}
            
            return {"text": text, "status": "success", "message": "Transcription successful."}
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True) # Log full traceback
            return {"text": None, "status": "error", "message": f"Error in transcription: {e}. Please ensure your internet connection is stable."}

    async def transcribe_audio_bytes(self, audio_bytes: bytes, language: str = None) -> dict:
        """Converts audio data from bytes to text using Whisper (asynchronous).
        Returns a dictionary with 'text' and 'status'.
        """
        if not audio_bytes:
            logger.error("Audio data is empty for bytes transcription.")
            return {"text": None, "status": "error", "message": "Audio data is empty."}
        
        try:
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav" # Required by OpenAI API for BytesIO
            response = await self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                response_format="text",
                language=language
            )
            text = response.strip()
            logger.info(f"Transcription from bytes: {text}")

            if not text:
                # For `transcribe_audio_bytes`, if no text is returned, it's safer to categorize
                # as a generic 'no speech' rather than trying to perform in-memory silence detection
                # unless explicitly required and implemented.
                return {"text": "", "status": "no_speech", "message": "No clear speech detected from the provided audio data."}
            
            return {"text": text, "status": "success", "message": "Transcription successful."}
        
        except Exception as e:
            logger.error(f"Transcription from bytes failed: {e}", exc_info=True) # Log full traceback
            return {"text": None, "status": "error", "message": f"Error in transcription from bytes: {e}"}

# === Example Usage ===
if __name__ == "__main__":
    async def main():
        # Configure logging for the example usage
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        logger.info("Starting STT module example...")
        stt_processor = STT()
        
        # Test Case 1: Standard recording
        logger.info("\n--- Testing Standard Recording ---")
        audio_path_1 = stt_processor.record_audio(duration=5)
        if audio_path_1:
            transcription_1 = await stt_processor.transcribe_audio(audio_path_1)
            logger.info(f"Transcription 1 Result: {transcription_1}")
            os.remove(audio_path_1)
        
        # Test Case 2: Silence (you'll need to stay silent for 5 seconds)
        logger.info("\n--- Testing Silence (stay silent for 5 seconds) ---")
        await asyncio.sleep(1) # Give time to prepare for silence
        audio_path_2 = stt_processor.record_audio(duration=5)
        if audio_path_2:
            transcription_2 = await stt_processor.transcribe_audio(audio_path_2)
            logger.info(f"Transcription 2 Result (Silence): {transcription_2}")
            os.remove(audio_path_2)
            
        # Test Case 3: Very short, unclear speech (try saying "uhm" very quietly)
        logger.info("\n--- Testing Unclear Speech (try mumbling briefly for 5 seconds) ---")
        await asyncio.sleep(1) # Give time to prepare
        audio_path_3 = stt_processor.record_audio(duration=5)
        if audio_path_3:
            transcription_3 = await stt_processor.transcribe_audio(audio_path_3)
            logger.info(f"Transcription 3 Result (Unclear Speech): {transcription_3}")
            os.remove(audio_path_3)
        
        logger.info("\nSTT module example finished.")

    asyncio.run(main())