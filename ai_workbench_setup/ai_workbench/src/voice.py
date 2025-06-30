import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import speech_recognition as sr
from gtts import gTTS
import io
import wave
from utils.logger import setup_logger
from utils.helpers import ensure_directory


class VoiceProcessor:
    """
    Enhanced voice processing system with speech-to-text and text-to-speech capabilities
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        
        # Load configuration
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            voice_config = config.get("voice", {})
            self.input_enabled = voice_config.get("input_enabled", True)
            self.output_enabled = voice_config.get("output_enabled", True)
            self.language = voice_config.get("language", "en")
            
        except Exception as e:
            self.logger.warning(f"Could not load voice configuration: {e}")
            self.input_enabled = True
            self.output_enabled = True
            self.language = "en"
        
        # Setup output directory
        self.output_dir = Path("data/voice_output")
        ensure_directory(self.output_dir)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Supported languages for different services
        self.supported_languages = {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            "ru": "Russian"
        }
        
        # Calibrate microphone
        self._calibrate_microphone()
        
        self.logger.info(f"Voice processor initialized (input: {self.input_enabled}, output: {self.output_enabled})")

    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.logger.info("Microphone calibrated for ambient noise")
        except Exception as e:
            self.logger.warning(f"Microphone calibration failed: {e}")

    def speech_to_text(self, audio_data: bytes, language: str = None) -> str:
        """
        Convert speech audio to text
        
        Args:
            audio_data: Audio data as bytes
            language: Language code (e.g., 'en', 'es')
            
        Returns:
            Transcribed text
        """
        if not self.input_enabled:
            raise RuntimeError("Speech-to-text is disabled")
        
        try:
            # Use provided language or default
            lang = language or self.language
            
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            try:
                # Load audio file
                with sr.AudioFile(tmp_file_path) as source:
                    audio = self.recognizer.record(source)
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio, language=lang)
                
                self.logger.info(f"Speech recognition successful: {text[:50]}...")
                return text
                
            except sr.UnknownValueError:
                self.logger.warning("Could not understand audio")
                return ""
            except sr.RequestError as e:
                self.logger.error(f"Speech recognition service error: {e}")
                return ""
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            self.logger.error(f"Error in speech-to-text: {e}")
            return ""

    def text_to_speech(self, text: str, language: str = None, slow: bool = False) -> Optional[str]:
        """
        Convert text to speech audio file
        
        Args:
            text: Text to convert to speech
            language: Language code
            slow: Whether to speak slowly
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.output_enabled:
            raise RuntimeError("Text-to-speech is disabled")
        
        try:
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Use provided language or default
            lang = language or self.language
            
            # Create TTS object
            tts = gTTS(text=text, lang=lang, slow=slow)
            
            # Generate unique filename
            timestamp = int(time.time())
            filename = f"speech_{timestamp}.mp3"
            file_path = self.output_dir / filename
            
            # Save audio file
            tts.save(str(file_path))
            
            self.logger.info(f"Text-to-speech generated: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {e}")
            return None

    def record_audio(self, duration: int = 5, sample_rate: int = 44100) -> Optional[bytes]:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
            
        Returns:
            Audio data as bytes or None if failed
        """
        if not self.input_enabled:
            raise RuntimeError("Audio recording is disabled")
        
        try:
            self.logger.info(f"Recording audio for {duration} seconds...")
            
            with self.microphone as source:
                # Record audio
                audio = self.recognizer.record(source, duration=duration)
                
                # Convert to bytes
                audio_data = audio.get_wav_data()
                
                self.logger.info(f"Audio recorded: {len(audio_data)} bytes")
                return audio_data
                
        except Exception as e:
            self.logger.error(f"Error recording audio: {e}")
            return None

    def process_voice_input(self, audio_data: bytes, language: str = None) -> Dict[str, Any]:
        """
        Process voice input and return structured result
        
        Args:
            audio_data: Audio data as bytes
            language: Language code
            
        Returns:
            Processing result with text and metadata
        """
        try:
            start_time = time.time()
            
            # Convert speech to text
            text = self.speech_to_text(audio_data, language)
            
            processing_time = time.time() - start_time
            
            result = {
                "text": text,
                "success": bool(text and text.strip()),
                "processing_time": processing_time,
                "language": language or self.language,
                "audio_size": len(audio_data),
                "timestamp": time.time()
            }
            
            if not result["success"]:
                result["error"] = "No speech detected or recognition failed"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing voice input: {e}")
            return {
                "text": "",
                "success": False,
                "error": str(e),
                "processing_time": 0.0,
                "language": language or self.language,
                "audio_size": len(audio_data) if audio_data else 0,
                "timestamp": time.time()
            }

    def generate_voice_response(self, text: str, language: str = None, 
                              auto_play: bool = False) -> Dict[str, Any]:
        """
        Generate voice response from text
        
        Args:
            text: Text to convert to speech
            language: Language code
            auto_play: Whether response should auto-play
            
        Returns:
            Response with audio file path and metadata
        """
        try:
            start_time = time.time()
            
            # Generate speech
            audio_path = self.text_to_speech(text, language)
            
            generation_time = time.time() - start_time
            
            result = {
                "audio_path": audio_path,
                "success": audio_path is not None,
                "generation_time": generation_time,
                "language": language or self.language,
                "text_length": len(text),
                "auto_play": auto_play,
                "timestamp": time.time()
            }
            
            if not result["success"]:
                result["error"] = "Speech generation failed"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating voice response: {e}")
            return {
                "audio_path": None,
                "success": False,
                "error": str(e),
                "generation_time": 0.0,
                "language": language or self.language,
                "text_length": len(text) if text else 0,
                "auto_play": auto_play,
                "timestamp": time.time()
            }

    def test_voice_system(self) -> Dict[str, Any]:
        """
        Test voice system functionality
        
        Returns:
            Test results
        """
        test_results = {
            "input_available": False,
            "output_available": False,
            "microphones": {},
            "languages_supported": len(self.supported_languages),
            "test_timestamp": time.time()
        }
        
        try:
            # Test microphone access
            if self.input_enabled:
                try:
                    # List available microphones
                    mic_list = sr.Microphone.list_microphone_names()
                    test_results["microphones"] = {i: name for i, name in enumerate(mic_list)}
                    test_results["input_available"] = len(mic_list) > 0
                    self.logger.info(f"Found {len(mic_list)} microphones")
                except Exception as e:
                    self.logger.warning(f"Microphone test failed: {e}")
            
            # Test text-to-speech
            if self.output_enabled:
                try:
                    test_text = "Voice system test"
                    test_path = self.text_to_speech(test_text)
                    test_results["output_available"] = test_path is not None
                    
                    # Clean up test file
                    if test_path and os.path.exists(test_path):
                        os.unlink(test_path)
                        
                    self.logger.info("Text-to-speech test successful")
                except Exception as e:
                    self.logger.warning(f"Text-to-speech test failed: {e}")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Voice system test failed: {e}")
            test_results["error"] = str(e)
            return test_results

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        return [
            {"code": code, "name": name} 
            for code, name in self.supported_languages.items()
        ]

    def cleanup_old_files(self, hours_old: int = 24) -> int:
        """
        Clean up old audio files
        
        Args:
            hours_old: Age threshold in hours
            
        Returns:
            Number of files deleted
        """
        try:
            from datetime import datetime, timedelta
            
            cutoff_time = datetime.now() - timedelta(hours=hours_old)
            deleted_count = 0
            
            for file_path in self.output_dir.glob("*.mp3"):
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    self.logger.debug(f"Could not delete {file_path}: {e}")
            
            self.logger.info(f"Cleaned up {deleted_count} old audio files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up audio files: {e}")
            return 0

    def get_voice_settings(self) -> Dict[str, Any]:
        """Get current voice settings"""
        return {
            "input_enabled": self.input_enabled,
            "output_enabled": self.output_enabled,
            "default_language": self.language,
            "supported_languages": self.supported_languages,
            "output_directory": str(self.output_dir)
        }

    def update_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Update voice settings
        
        Args:
            settings: New settings
            
        Returns:
            Success status
        """
        try:
            if "input_enabled" in settings:
                self.input_enabled = bool(settings["input_enabled"])
            
            if "output_enabled" in settings:
                self.output_enabled = bool(settings["output_enabled"])
            
            if "language" in settings:
                lang = settings["language"]
                if lang in self.supported_languages:
                    self.language = lang
                else:
                    self.logger.warning(f"Unsupported language: {lang}")
            
            self.logger.info("Voice settings updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating voice settings: {e}")
            return False