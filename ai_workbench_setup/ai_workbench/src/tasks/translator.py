from src.models.base_model import BaseModel
from utils.logger import setup_logger
from utils.helpers import validate_text, validate_model_params
from typing import List, Dict, Any, Optional
import time

class Translator:
    """
    Translator class that handles text translation across multiple models
    """
    
    # Supported languages mapping
    SUPPORTED_LANGUAGES = {
        "spanish": "Spanish",
        "french": "French", 
        "german": "German",
        "italian": "Italian",
        "portuguese": "Portuguese",
        "chinese": "Chinese",
        "japanese": "Japanese",
        "korean": "Korean",
        "arabic": "Arabic",
        "hindi": "Hindi",
        "russian": "Russian",
        "dutch": "Dutch",
        "polish": "Polish",
        "swedish": "Swedish",
        "norwegian": "Norwegian",
        "danish": "Danish"
    }
    
    def __init__(self, models: List[BaseModel]):
        self.logger = setup_logger(__name__)
        
        # Filter only available models
        self.models = [model for model in models if model.is_available()]
        
        if not self.models:
            self.logger.warning("No available models for translation")
        else:
            model_names = [model.get_name() for model in self.models]
            self.logger.info(f"Translator initialized with models: {model_names}")

    def translate(self, text: str, target_lang: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Translate text using all available models
        
        Args:
            text: Text to translate
            target_lang: Target language
            params: Generation parameters
            
        Returns:
            List of result dictionaries with model outputs
        """
        try:
            # Validate inputs
            text = validate_text(text)
            target_lang = self._normalize_language(target_lang)
            params = validate_model_params(params)
            
            self.logger.info(f"Starting translation to {target_lang} for {len(self.models)} models")
            self.logger.debug(f"Input text length: {len(text)} characters")
            self.logger.debug(f"Parameters: {params}")
            
            results = []
            
            for model in self.models:
                model_result = self._translate_with_model(model, text, target_lang, params)
                results.append(model_result)
            
            # Log summary statistics
            successful_results = [r for r in results if r.get("output") is not None]
            self.logger.info(
                f"Translation completed: {len(successful_results)}/{len(self.models)} models succeeded"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in translation: {e}")
            raise

    def _translate_with_model(self, model: BaseModel, text: str, target_lang: str, params: Dict[str, Any]) -> Dict:
        """
        Translate text with a single model
        
        Args:
            model: Model instance
            text: Text to translate
            target_lang: Target language
            params: Generation parameters
            
        Returns:
            Result dictionary
        """
        model_name = model.get_name()
        start_time = time.time()
        
        try:
            self.logger.debug(f"Starting translation with {model_name}")
            
            # Check if model is available
            if not model.is_available():
                raise RuntimeError(f"Model {model_name} is not available")
            
            # Check input length
            if not model.check_input_length(text):
                self.logger.warning(f"Input text may be too long for {model_name}")
            
            # Generate translation
            translation, inference_time = model.translate(text, target_lang, params)
            
            # Validate output
            if not translation or not isinstance(translation, str):
                raise ValueError("Model returned empty or invalid translation")
            
            # Only basic cleaning - preserve the actual text structure
            translation = translation.strip()
            
            # Detect potential translation quality issues
            quality_issues = self._detect_quality_issues(text, translation, target_lang)
            
            result = {
                "model": model_name,
                "output": translation,  # Don't apply aggressive cleaning here
                "inference_time": inference_time,
                "success": True,
                "target_language": target_lang,
                "word_count": len(translation.split()),
                "character_count": len(translation),
                "quality_issues": quality_issues
            }
            
            self.logger.info(
                f"✓ {model_name}: {result['word_count']} words in {inference_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"✗ {model_name} failed: {error_msg}")
            
            return {
                "model": model_name,
                "output": None,
                "inference_time": total_time,
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__,
                "target_language": target_lang
            }

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language name to standard format
        
        Args:
            language: Input language name
            
        Returns:
            Normalized language name
        """
        if not language:
            raise ValueError("Target language cannot be empty")
        
        language_lower = language.lower().strip()
        
        # Direct match
        if language_lower in self.SUPPORTED_LANGUAGES:
            return self.SUPPORTED_LANGUAGES[language_lower]
        
        # Check if it's already in proper format
        if language.title() in self.SUPPORTED_LANGUAGES.values():
            return language.title()
        
        # Fuzzy matching for common variations
        language_variations = {
            "esp": "Spanish",
            "spa": "Spanish", 
            "es": "Spanish",
            "fr": "French",
            "fra": "French",
            "de": "German",
            "deu": "German",
            "ger": "German",
            "it": "Italian",
            "ita": "Italian",
            "pt": "Portuguese",
            "por": "Portuguese",
            "zh": "Chinese",
            "chi": "Chinese",
            "ja": "Japanese",
            "jpn": "Japanese",
            "ko": "Korean",
            "kor": "Korean",
            "ar": "Arabic",
            "ara": "Arabic",
            "hi": "Hindi",
            "hin": "Hindi",
            "ru": "Russian",
            "rus": "Russian",
            "nl": "Dutch",
            "nld": "Dutch",
            "pl": "Polish",
            "pol": "Polish",
            "sv": "Swedish",
            "swe": "Swedish",
            "no": "Norwegian",
            "nor": "Norwegian",
            "da": "Danish",
            "dan": "Danish"
        }
        
        if language_lower in language_variations:
            return language_variations[language_lower]
        
        # If no match found, return as-is with proper capitalization
        self.logger.warning(f"Unknown language: {language}. Using as-is.")
        return language.title()

    def _detect_quality_issues(self, original: str, translation: str, target_lang: str) -> List[str]:
        """
        Detect potential quality issues in translation
        
        Args:
            original: Original text
            translation: Translated text
            target_lang: Target language
            
        Returns:
            List of detected issues
        """
        issues = []
        
        try:
            # Basic checks
            if not translation.strip():
                issues.append("Empty translation")
                return issues
            
            if original.lower() == translation.lower():
                issues.append("Translation identical to original")
            
            # Length checks
            orig_words = len(original.split())
            trans_words = len(translation.split())
            
            if trans_words > orig_words * 3:
                issues.append("Translation much longer than original")
            elif trans_words < orig_words * 0.3:
                issues.append("Translation much shorter than original")
            
            # Check for untranslated English words (basic heuristic)
            if target_lang.lower() not in ["english"]:
                english_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
                trans_words_lower = set(translation.lower().split())
                common_english = english_words.intersection(trans_words_lower)
                
                if len(common_english) > 3:
                    issues.append("Contains many English words")
            
            # Check for repeated phrases
            words = translation.split()
            if len(words) > 4:
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i+3])
                    if translation.count(phrase) > 2:
                        issues.append("Contains repeated phrases")
                        break
            
        except Exception as e:
            self.logger.debug(f"Error in quality detection: {e}")
        
        return issues

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return list(self.SUPPORTED_LANGUAGES.values())

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return [model.get_name() for model in self.models if model.is_available()]

    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about all models"""
        return [model.get_model_info() for model in self.models]