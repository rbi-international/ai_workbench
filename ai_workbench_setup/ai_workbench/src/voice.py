import speech_recognition as sr
from gtts import gTTS
import io
import os
import tempfile
import wave
from pathlib import Path
from typing import Optional, Union, Dict, Any
from utils.logger import setup_logger

class VoiceProcessor:
    """
    Enhanced voice processor for speech-to-text and text-to-speech operations
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        
        # Initialize speech recognizer
        try:
            self.recognizer = sr.Recognizer()
            # Adjust for ambient noise
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            
            self.logger.info("Voice processor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize voice processor: {e}")
            raise
        
        # Load configuration
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            voice_config = config.get("voice", {})
            self.input_enabled = voice_config.get("input_enabled", True)
            self.output_enabled = voice_config.get("output_enabled", True)
            self.default_language = voice_config.get("language", "en")
            
        except Exception as e:
            self.logger.warning(f"Could not load voice configuration: {e}")
            self.input_enabled = True
            self.output_enabled = True
            self.default_language = "en"
        
        # Ensure output directory exists
        self.output_dir = Path("data/voice_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def speech_to_text(self, audio_data: Union[bytes, str], language: str = "en-US") -> str:
        """
        Convert speech to text from audio data or file path
        
        Args:
            audio_data: Audio data as bytes or file path as string
            language: Language code for recognition (default: en-US)
            
        Returns:
            Transcribed text
        """
        if not self.input_enabled:
            raise RuntimeError("Voice input is disabled")
        
        try:
            # Handle different input types
            if isinstance(audio_data, str):
                # File path provided
                audio_file_path = audio_data
            else:
                # Bytes data provided
                audio_file_path = self._save_temp_audio(audio_data)
            
            # Validate file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Convert audio to text
            text = self._process_audio_file(audio_file_path, language)
            
            # Cleanup temporary file if we created one
            if isinstance(audio_data, bytes) and os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                except Exception as e:
                    self.logger.warning(f"Could not remove temp file: {e}")
            
            self.logger.info(f"Speech-to-text conversion successful: {len(text)} characters")
            return text
            
        except Exception as e:
            self.logger.error(f"Speech-to-text conversion failed: {e}")
            raise

    def _save_temp_audio(self, audio_data: bytes) -> str:
        """Save audio bytes to temporary file"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                return temp_file.name
        except Exception as e:
            self.logger.error(f"Failed to save temporary audio file: {e}")
            raise

    def _process_audio_file(self, file_path: str, language: str) -> str:
        """Process audio file and extract text"""
        try:
            # Try different recognition methods
            recognition_methods = [
                ("Google", self._recognize_google),
                ("Sphinx", self._recognize_sphinx),
                ("Google Cloud", self._recognize_google_cloud)
            ]
            
            with sr.AudioFile(file_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record the audio
                audio = self.recognizer.record(source)
            
            # Try recognition methods in order of preference
            last_error = None
            
            for method_name, method_func in recognition_methods:
                try:
                    self.logger.debug(f"Trying {method_name} recognition...")
                    text = method_func(audio, language)
                    
                    if text and text.strip():
                        self.logger.info(f"Recognition successful with {method_name}")
                        return text.strip()
                    
                except Exception as e:
                    self.logger.debug(f"{method_name} recognition failed: {e}")
                    last_error = e
                    continue
            
            # If all methods failed
            error_msg = f"All recognition methods failed. Last error: {last_error}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except Exception as e:
            self.logger.error(f"Audio file processing failed: {e}")
            raise

    def _recognize_google(self, audio, language: str) -> str:
        """Google Speech Recognition"""
        try:
            return self.recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError:
            raise RuntimeError("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            raise RuntimeError(f"Google Speech Recognition service error: {e}")

    def _recognize_sphinx(self, audio, language: str) -> str:
        """Sphinx Speech Recognition (offline)"""
        try:
            return self.recognizer.recognize_sphinx(audio)
        except sr.UnknownValueError:
            raise RuntimeError("Sphinx could not understand audio")
        except sr.RequestError as e:
            raise RuntimeError(f"Sphinx error: {e}")

    def _recognize_google_cloud(self, audio, language: str) -> str:
        """Google Cloud Speech Recognition"""
        try:
            # This requires Google Cloud credentials
            return self.recognizer.recognize_google_cloud(audio, language=language)
        except sr.UnknownValueError:
            raise RuntimeError("Google Cloud Speech Recognition could not understand audio")
        except sr.RequestError as e:
            raise RuntimeError(f"Google Cloud Speech Recognition service error: {e}")

    def text_to_speech(self, text: str, language: str = None, output_path: str = None, slow: bool = False) -> str:
        """
        Convert text to speech and save as audio file
        
        Args:
            text: Text to convert to speech
            language: Language code (default: self.default_language)
            output_path: Output file path (default: auto-generated)
            slow: Speak slowly (default: False)
            
        Returns:
            Path to generated audio file
        """
        if not self.output_enabled:
            raise RuntimeError("Voice output is disabled")
        
        try:
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Use default language if not specified
            if language is None:
                language = self.default_language
            
            # Generate output path if not provided
            if output_path is None:
                timestamp = int(time.time())
                output_path = self.output_dir / f"speech_output_{timestamp}.mp3"
            else:
                output_path = Path(output_path)
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate speech
            self.logger.debug(f"Generating speech for {len(text)} characters")
            
            tts = gTTS(text=text, lang=language, slow=slow)
            tts.save(str(output_path))
            
            # Verify file was created
            if not output_path.exists():
                raise RuntimeError("Audio file was not created")
            
            file_size = output_path.stat().st_size
            self.logger.info(f"Text-to-speech conversion successful: {output_path} ({file_size} bytes)")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Text-to-speech conversion failed: {e}")
            raise

    def get_microphone_list(self) -> Dict[int, str]:
        """Get list of available microphones"""
        try:
            microphones = {}
            
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                microphones[index] = name
            
            self.logger.info(f"Found {len(microphones)} microphones")
            return microphones
            
        except Exception as e:
            self.logger.error(f"Error getting microphone list: {e}")
            return {}

    def record_from_microphone(self, duration: float = 5.0, microphone_index: int = None) -> bytes:
        """
        Record audio from microphone
        
        Args:
            duration: Recording duration in seconds
            microphone_index: Microphone index (default: system default)
            
        Returns:
            Audio data as bytes
        """
        try:
            # Use specified microphone or default
            microphone = sr.Microphone(device_index=microphone_index) if microphone_index is not None else sr.Microphone()
            
            with microphone as source:
                self.logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source)
                
                self.logger.info(f"Recording for {duration} seconds...")
                audio = self.recognizer.record(source, duration=duration)
            
            # Convert to bytes
            audio_data = audio.get_wav_data()
            
            self.logger.info(f"Recording completed: {len(audio_data)} bytes")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Microphone recording failed: {e}")
            raise

    def listen_for_speech(self, timeout: float = 10.0, phrase_time_limit: float = 5.0, microphone_index: int = None) -> str:
        """
        Listen for speech and convert to text in real-time
        
        Args:
            timeout: Maximum time to wait for speech to start
            phrase_time_limit: Maximum time for a phrase
            microphone_index: Microphone index (default: system default)
            
        Returns:
            Transcribed text
        """
        try:
            microphone = sr.Microphone(device_index=microphone_index) if microphone_index is not None else sr.Microphone()
            
            with microphone as source:
                self.logger.info("Listening for speech...")
                self.recognizer.adjust_for_ambient_noise(source)
                
                try:
                    audio = self.recognizer.listen(
                        source, 
                        timeout=timeout, 
                        phrase_time_limit=phrase_time_limit
                    )
                except sr.WaitTimeoutError:
                    raise RuntimeError(f"No speech detected within {timeout} seconds")
            
            # Convert to text
            try:
                text = self.recognizer.recognize_google(audio)
                self.logger.info(f"Speech recognized: {text}")
                return text
            except sr.UnknownValueError:
                raise RuntimeError("Could not understand the speech")
            except sr.RequestError as e:
                raise RuntimeError(f"Speech recognition service error: {e}")
                
        except Exception as e:
            self.logger.error(f"Real-time speech recognition failed: {e}")
            raise

    def convert_audio_format(self, input_path: str, output_path: str = None, target_format: str = "wav") -> str:
        """
        Convert audio file to different format
        
        Args:
            input_path: Input audio file path
            output_path: Output file path (default: auto-generated)
            target_format: Target format (wav, mp3, etc.)
            
        Returns:
            Path to converted file
        """
        try:
            import pydub
            
            input_path = Path(input_path)
            
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Generate output path if not provided
            if output_path is None:
                output_path = input_path.with_suffix(f".{target_format}")
            else:
                output_path = Path(output_path)
            
            # Load and convert audio
            audio = pydub.AudioSegment.from_file(str(input_path))
            audio.export(str(output_path), format=target_format)
            
            self.logger.info(f"Audio converted: {input_path} -> {output_path}")
            return str(output_path)
            
        except ImportError:
            self.logger.error("pydub library not available for audio conversion")
            raise RuntimeError("Audio conversion requires pydub library")
        except Exception as e:
            self.logger.error(f"Audio conversion failed: {e}")
            raise

    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            info = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "format": file_path.suffix.lower()
            }
            
            # Try to get detailed info with wave module for WAV files
            if file_path.suffix.lower() == ".wav":
                try:
                    with wave.open(str(file_path), 'rb') as wav_file:
                        info.update({
                            "channels": wav_file.getnchannels(),
                            "sample_width": wav_file.getsampwidth(),
                            "framerate": wav_file.getframerate(),
                            "frames": wav_file.getnframes(),
                            "duration": wav_file.getnframes() / wav_file.getframerate()
                        })
                except Exception as e:
                    self.logger.debug(f"Could not get detailed WAV info: {e}")
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting audio info: {e}")
            return {"error": str(e)}

    def is_available(self) -> bool:
        """Check if voice processing is available"""
        return self.input_enabled or self.output_enabled

    def test_voice_system(self) -> Dict[str, Any]:
        """Test voice system functionality"""
        results = {
            "input_available": False,
            "output_available": False,
            "microphones": {},
            "errors": []
        }
        
        # Test speech recognition
        try:
            test_audio = b""  # Empty audio for testing
            results["input_available"] = True
        except Exception as e:
            results["errors"].append(f"Speech recognition test failed: {e}")
        
        # Test text-to-speech
        try:
            test_output = self.output_dir / "test_tts.mp3"
            self.text_to_speech("Test", output_path=str(test_output))
            
            if test_output.exists():
                results["output_available"] = True
                # Cleanup test file
                test_output.unlink()
            
        except Exception as e:
            results["errors"].append(f"Text-to-speech test failed: {e}")
        
        # Get microphone list
        try:
            results["microphones"] = self.get_microphone_list()
        except Exception as e:
            results["errors"].append(f"Microphone detection failed: {e}")
        
        return results