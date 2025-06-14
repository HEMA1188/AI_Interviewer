# tts.py
import io
import tempfile
import pygame
import logging
import sys
import os
import asyncio
from typing import Union

import pyttsx3
from gtts import gTTS

# === Setup Project Path (adjust if needed) ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.config import client, TTS_ENGINE, DEFAULT_VOICE, TTS_PROVIDER

# === Logging ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TTS:
    def __init__(self, engine=TTS_ENGINE):
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        try:
            pygame.mixer.init()
        except pygame.error as e:
            self.logger.error(f"Pygame mixer initialization failed: {e}")
            self.mixer_initialized = False
        else:
            self.mixer_initialized = True

    def __del__(self):
        if self.mixer_initialized:
            pygame.mixer.quit()

    async def play_audio_file(self, audio_path: str):
        """Plays an audio file using pygame (asynchronous)."""
        if self.mixer_initialized and os.path.exists(audio_path):
            try:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    await asyncio.sleep(0.1)
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except pygame.error as e:
                self.logger.error(f"Pygame audio playback error: {e}")
        else:
            self.logger.warning("Pygame mixer not initialized or audio file not found.")

    async def text_to_speech(self, text: str, output_path: Union[str, None] = None) -> str:
        """Converts text to speech using the selected TTS provider (asynchronous)."""
        try:
            if TTS_PROVIDER == "openai":
                return await self.text_to_speech_openai(text, output_path)
            elif TTS_PROVIDER == "pyttsx3":
                return await self._run_sync_in_executor(self.text_to_speech_pyttsx3, text, output_path)
            elif TTS_PROVIDER == "gtts":
                return await self._run_sync_in_executor(self.text_to_speech_gtts, text, output_path)
            else:
                self.logger.error("Invalid TTS provider selected.")
                return ""
        except Exception as e:
            self.logger.error(f"TTS conversion failed: {e}")
            return ""

    async def text_to_speech_openai(self, text: str, output_path: Union[str, None] = None) -> str:
        """Converts text to speech using OpenAI's TTS API (asynchronous)."""
        tmp_path = output_path
        try:
            if not text:
                self.logger.error("TTS Error: 'text' is required")
                return None

            speech_text = text.strip()
            self.logger.info(f"Generating OpenAI TTS for: {speech_text}")

            response = await client.audio.speech.create(
                model=self.engine,
                voice=DEFAULT_VOICE,
                input=speech_text,
                response_format="mp3"
            )

            # Correctly read the audio stream
            audio_stream = await response.aread()
            if not tmp_path:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(audio_stream)
                    tmp_path = tmp.name
            else:
                with open(tmp_path, 'wb') as f:
                    f.write(audio_stream)

            self.logger.info(f"OpenAI TTS audio saved to {tmp_path}")
            await self.play_audio_file(tmp_path)
            return tmp_path

        except Exception as e:
            self.logger.error(f"OpenAI TTS conversion failed: {e}")
            return None
        finally:
            if output_path is None and tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def text_to_speech_pyttsx3(self, text: str, output_path: Union[str, None] = None) -> str:
        """Converts text to speech using pyttsx3 (synchronous)."""
        try:
            engine = pyttsx3.init()
            final_output_path = output_path or os.path.join(os.path.abspath("storage/audio"), "pyttsx3_output.mp3")
            engine.save_to_file(text, final_output_path)
            engine.runAndWait()
            self.logger.info(f"pyttsx3 TTS audio saved to {final_output_path}")
            return final_output_path
        except Exception as e:
            self.logger.error(f"pyttsx3 TTS conversion failed: {e}")
            return ""

    def text_to_speech_gtts(self, text: str, output_path: Union[str, None] = None) -> str:
        """Converts text to speech using gTTS (synchronous)."""
        try:
            tts = gTTS(text=text, lang="en")
            final_output_path = output_path or os.path.join(os.path.abspath("storage/audio"), "gtts_output.mp3")
            tts.save(final_output_path)
            self.logger.info(f"gTTS audio saved to {final_output_path}")
            return final_output_path
        except Exception as e:
            self.logger.error(f"gTTS TTS conversion failed: {e}")
            return ""

    @staticmethod
    async def _run_sync_in_executor(func, *args, **kwargs):
        """Runs a synchronous function in the asyncio event loop's executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
    # Inside your TTS class in audio/tts.py
    async def shutdown(self):
        """Placeholder for shutting down TTS resources."""
        # Add any necessary cleanup logic here
        print("TTS shutdown called.")
        pass

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)

    async def test_tts():
        tts_engine = TTS()
        
        text_to_speak_openai = "Hello, this is a test of OpenAI's text-to-speech."
        openai_path = await tts_engine.text_to_speech(text_to_speak_openai, output_path="openai_test.mp3")
        print(f"OpenAI TTS saved to: {openai_path}")

        text_to_speak_pyttsx3 = "This is a test using pyttsx3."
        pyttsx3_path = await tts_engine.text_to_speech(text_to_speak_pyttsx3, output_path="pyttsx3_test.mp3")
        print(f"pyttsx3 TTS saved to: {pyttsx3_path}")

        text_to_speak_gtts = "Testing Google Text-to-Speech."
        gtts_path = await tts_engine.text_to_speech(text_to_speak_gtts, output_path="gtts_test.mp3")
        print(f"gTTS saved to: {gtts_path}")

    asyncio.run(test_tts())
