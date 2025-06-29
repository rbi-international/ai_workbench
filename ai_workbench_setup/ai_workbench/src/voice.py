import speech_recognition as sr
from gtts import gTTS
import io
import os
from utils.logger import setup_logger

class VoiceProcessor:
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.recognizer = sr.Recognizer()

    def speech_to_text(self, audio_file: bytes) -> str:
        try:
            with io.BytesIO(audio_file) as f:
                audio = sr.AudioFile(f)
                with audio as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    self.logger.info("Converted speech to text")
                    return text
        except Exception as e:
            self.logger.error(f"Error in speech-to-text: {str(e)}")
            raise

    def text_to_speech(self, text: str, output_path: str = "output.mp3") -> str:
        try:
            tts = gTTS(text=text, lang="en")
            tts.save(output_path)
            self.logger.info(f"Generated speech at {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {str(e)}")
            raise