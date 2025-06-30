import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import io
from utils.logger import setup_logger
from utils.helpers import ensure_directory

# Try to import speech recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

# Try to import gTTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

class VoiceProcessor:
    """
    Enhanced voice processing system with better error handling
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
        
        # Initialize components if available
        self.recognizer = None
        self.microphone = None
        
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self._calibrate_microphone()
                self.logger.info("Speech recognition initialized successfully")
            except Exception as e:
                self.logger.warning(f"Speech recognition setup failed: {e}")
                self.input_enabled = False
        else:
            self.logger.warning("speech_recognition library not available")
            self.input_enabled = False
        
        if not GTTS_AVAILABLE:
            self.logger.warning("gTTS library not available")
            self.output_enabled = False
        
        # Supported languages
        self.supported_languages = {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese"
        }
        
        self.logger.info(f"Voice processor initialized (input: {self.input_enabled}, output: {self.output_enabled})")

    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        try:
            if self.microphone and self.recognizer:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.logger.info("Microphone calibrated for ambient noise")
        except Exception as e:
            self.logger.warning(f"Microphone calibration failed: {e}")

    def speech_to_text(self, audio_data: bytes, language: str = None) -> str:
        """
        Convert speech audio to text with better error handling
        
        Args:
            audio_data: Audio data as bytes
            language: Language code (e.g., 'en', 'es')
            
        Returns:
            Transcribed text
        """
        if not self.input_enabled or not self.recognizer:
            self.logger.error("Speech-to-text is not available")
            return ""
        
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
                
                # Try Google Speech Recognition first
                try:
                    text = self.recognizer.recognize_google(audio, language=lang)
                    self.logger.info(f"Speech recognition successful: {text[:50]}...")
                    return text
                except sr.UnknownValueError:
                    self.logger.warning("Google SR: Could not understand audio")
                    
                    # Fallback: try with different settings
                    try:
                        text = self.recognizer.recognize_google(audio, language="en-US")
                        self.logger.info(f"Fallback speech recognition successful: {text[:50]}...")
                        return text
                    except:
                        pass
                
                # If Google fails, try other recognizers
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    self.logger.info(f"Sphinx recognition successful: {text[:50]}...")
                    return text
                except:
                    pass
                
                self.logger.warning("All speech recognition methods failed")
                return ""
                
            except sr.RequestError as e:
                self.logger.error(f"Speech recognition service error: {e}")
                return ""
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
                    
        except Exception as e:
            self.logger.error(f"Error in speech-to-text: {e}")
            return ""

    def text_to_speech(self, text: str, language: str = None, slow: bool = False) -> Optional[str]:
        """
        Convert text to speech audio file with better error handling
        
        Args:
            text: Text to convert to speech
            language: Language code
            slow: Whether to speak slowly
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.output_enabled or not GTTS_AVAILABLE:
            self.logger.error("Text-to-speech is not available")
            return None
        
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

    def test_voice_system(self) -> Dict[str, Any]:
        """
        Test voice system functionality with detailed results
        
        Returns:
            Test results
        """
        test_results = {
            "input_available": False,
            "output_available": False,
            "microphones": {},
            "languages_supported": len(self.supported_languages),
            "test_timestamp": time.time(),
            "errors": []
        }
        
        try:
            # Test microphone access
            if self.input_enabled and SPEECH_RECOGNITION_AVAILABLE:
                try:
                    # List available microphones
                    mic_list = sr.Microphone.list_microphone_names()
                    test_results["microphones"] = {i: name for i, name in enumerate(mic_list)}
                    test_results["input_available"] = len(mic_list) > 0
                    self.logger.info(f"Found {len(mic_list)} microphones")
                except Exception as e:
                    test_results["errors"].append(f"Microphone test failed: {e}")
                    self.logger.warning(f"Microphone test failed: {e}")
            else:
                test_results["errors"].append("Speech recognition not available")
            
            # Test text-to-speech
            if self.output_enabled and GTTS_AVAILABLE:
                try:
                    test_text = "Voice system test"
                    test_path = self.text_to_speech(test_text)
                    test_results["output_available"] = test_path is not None
                    
                    # Clean up test file
                    if test_path and os.path.exists(test_path):
                        try:
                            os.unlink(test_path)
                        except:
                            pass
                        
                    self.logger.info("Text-to-speech test successful")
                except Exception as e:
                    test_results["errors"].append(f"Text-to-speech test failed: {e}")
                    self.logger.warning(f"Text-to-speech test failed: {e}")
            else:
                test_results["errors"].append("Text-to-speech not available")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Voice system test failed: {e}")
            test_results["errors"].append(str(e))
            return test_results
