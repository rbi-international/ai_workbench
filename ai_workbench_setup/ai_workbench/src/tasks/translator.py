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
            
            # Detect potential translation quality issues
            quality_issues = self._detect_quality_issues(text, translation, target_lang)
            
            result = {
                "model": model_name,
                "output": translation.strip(),
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

    def translate_with_specific_model(self, text: str, target_lang: str, model_name: str, params: Dict[str, Any]) -> Dict:
        """
        Translate text with a specific model
        
        Args:
            text: Text to translate
            target_lang: Target language
            model_name: Name of the model to use
            params: Generation parameters
            
        Returns:
            Result dictionary
        """
        try:
            # Find the model
            target_model = None
            for model in self.models:
                if model.get_name() == model_name:
                    target_model = model
                    break
            
            if target_model is None:
                raise ValueError(f"Model {model_name} not found or not available")
            
            # Validate inputs
            text = validate_text(text)
            target_lang = self._normalize_language(target_lang)
            params = validate_model_params(params)
            
            # Generate translation
            return self._translate_with_model(target_model, text, target_lang, params)
            
        except Exception as e:
            self.logger.error(f"Error in model-specific translation: {e}")
            raise

    def batch_translate(self, texts: List[str], target_lang: str, params: Dict[str, Any]) -> List[List[Dict]]:
        """
        Translate multiple texts
        
        Args:
            texts: List of texts to translate
            target_lang: Target language
            params: Generation parameters
            
        Returns:
            List of result lists (one per input text)
        """
        try:
            target_lang = self._normalize_language(target_lang)
            self.logger.info(f"Starting batch translation for {len(texts)} texts to {target_lang}")
            
            results = []
            for i, text in enumerate(texts):
                self.logger.debug(f"Processing text {i+1}/{len(texts)}")
                text_results = self.translate(text, target_lang, params)
                results.append(text_results)
            
            self.logger.info("Batch translation completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch translation: {e}")
            raise

    def compare_translations(self, text: str, target_lang: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate translations and provide comparison metrics
        
        Args:
            text: Text to translate
            target_lang: Target language
            params: Generation parameters
            
        Returns:
            Comparison results
        """
        try:
            # Generate translations
            results = self.translate(text, target_lang, params)
            
            # Calculate comparison metrics
            successful_results = [r for r in results if r.get("output") is not None]
            
            if not successful_results:
                return {"error": "No successful translations generated"}
            
            # Basic statistics
            word_counts = [r["word_count"] for r in successful_results]
            inference_times = [r["inference_time"] for r in successful_results]
            quality_scores = [len(r.get("quality_issues", [])) for r in successful_results]
            
            comparison = {
                "translations": results,
                "target_language": target_lang,
                "statistics": {
                    "total_models": len(results),
                    "successful_models": len(successful_results),
                    "average_word_count": sum(word_counts) / len(word_counts),
                    "average_inference_time": sum(inference_times) / len(inference_times),
                    "fastest_model": min(successful_results, key=lambda x: x["inference_time"])["model"],
                    "shortest_translation": min(successful_results, key=lambda x: x["word_count"])["model"],
                    "longest_translation": max(successful_results, key=lambda x: x["word_count"])["model"],
                    "best_quality": min(successful_results, key=lambda x: len(x.get("quality_issues", [])))["model"]
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error in translation comparison: {e}")
            raise

    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Simple language detection (basic heuristic)
        
        Args:
            text: Text to analyze
            
        Returns:
            Language detection results
        """
        try:
            if not text or not text.strip():
                return {"error": "Empty text provided"}
            
            # Basic character-based detection
            text_lower = text.lower()
            
            # Common word patterns for different languages
            language_indicators = {
                "Spanish": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para"],
                "French": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour", "dans", "ce", "son", "une", "sur", "avec", "ne"],
                "German": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "für", "ist", "im", "dem", "nicht", "ein"],
                "Italian": ["il", "di", "che", "e", "la", "per", "un", "in", "con", "del", "da", "non", "al", "le", "si", "gli", "come", "più", "lo"],
                "Portuguese": ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por"],
            }
            
            scores = {}
            words = text_lower.split()
            
            if len(words) < 3:
                return {"detected_language": "Unknown", "confidence": 0.0, "reason": "Text too short"}
            
            for language, indicators in language_indicators.items():
                score = sum(1 for word in words if word in indicators)
                scores[language] = score / len(words)
            
            # Find best match
            if scores:
                best_language = max(scores, key=scores.get)
                confidence = scores[best_language]
                
                if confidence > 0.1:
                    return {
                        "detected_language": best_language,
                        "confidence": confidence,
                        "all_scores": scores
                    }
            
            return {"detected_language": "Unknown", "confidence": 0.0, "all_scores": scores}
            
        except Exception as e:
            self.logger.error(f"Error in language detection: {e}")
            return {"error": str(e)}

    def validate_translation_quality(self, original: str, translation: str, target_lang: str) -> Dict[str, Any]:
        """
        Comprehensive validation of translation quality
        
        Args:
            original: Original text
            translation: Translated text
            target_lang: Target language
            
        Returns:
            Quality assessment
        """
        try:
            if not original or not translation:
                return {"error": "Empty original or translation text"}
            
            # Basic metrics
            original_words = len(original.split())
            translation_words = len(translation.split())
            length_ratio = translation_words / original_words if original_words > 0 else 0
            
            # Detect issues
            issues = self._detect_quality_issues(original, translation, target_lang)
            
            # Calculate quality score
            quality_score = self._calculate_translation_quality_score(
                original, translation, target_lang, issues, length_ratio
            )
            
            # Categorize quality
            if quality_score >= 0.8:
                quality_category = "Excellent"
            elif quality_score >= 0.6:
                quality_category = "Good"
            elif quality_score >= 0.4:
                quality_category = "Fair"
            elif quality_score >= 0.2:
                quality_category = "Poor"
            else:
                quality_category = "Very Poor"
            
            return {
                "quality_score": quality_score,
                "quality_category": quality_category,
                "original_words": original_words,
                "translation_words": translation_words,
                "length_ratio": length_ratio,
                "issues": issues,
                "target_language": target_lang,
                "recommendations": self._get_quality_recommendations(issues, length_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"Error in translation quality validation: {e}")
            return {"error": str(e)}

    def _calculate_translation_quality_score(self, original: str, translation: str, target_lang: str, issues: List[str], length_ratio: float) -> float:
        """Calculate translation quality score"""
        base_score = 1.0
        
        # Penalize for issues
        base_score -= len(issues) * 0.15
        
        # Penalize for extreme length differences
        if length_ratio > 2.5 or length_ratio < 0.4:
            base_score -= 0.2
        elif length_ratio > 2.0 or length_ratio < 0.5:
            base_score -= 0.1
        
        # Bonus for reasonable length
        if 0.7 <= length_ratio <= 1.5:
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))

    def _get_quality_recommendations(self, issues: List[str], length_ratio: float) -> List[str]:
        """Get recommendations based on quality issues"""
        recommendations = []
        
        if "Empty translation" in issues:
            recommendations.append("The translation appears to be empty. Check if the model supports the target language.")
        
        if "Translation identical to original" in issues:
            recommendations.append("The translation is identical to the original. The model may not have understood the translation request.")
        
        if "Translation much longer than original" in issues:
            recommendations.append("The translation is significantly longer than expected. Consider using more specific prompts.")
        
        if "Translation much shorter than original" in issues:
            recommendations.append("The translation may be incomplete. Try adjusting the max_tokens parameter.")
        
        if "Contains many English words" in issues:
            recommendations.append("The translation contains many English words. The model may not be fully translating to the target language.")
        
        if "Contains repeated phrases" in issues:
            recommendations.append("The translation has repetitive content. Try adjusting temperature or repetition penalties.")
        
        if length_ratio > 2.0:
            recommendations.append("Consider using a more concise translation approach or check if the model is adding explanations.")
        elif length_ratio < 0.5:
            recommendations.append("The translation may be too brief. Consider requesting a more complete translation.")
        
        if not recommendations:
            recommendations.append("Translation quality appears good based on basic metrics.")
        
        return recommendations

    def multi_language_translate(self, text: str, target_languages: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate text to multiple languages simultaneously
        
        Args:
            text: Text to translate
            target_languages: List of target languages
            params: Generation parameters
            
        Returns:
            Dictionary with translations for each language
        """
        try:
            text = validate_text(text)
            params = validate_model_params(params)
            
            results = {}
            
            for lang in target_languages:
                try:
                    lang_results = self.translate(text, lang, params)
                    results[lang] = lang_results
                    self.logger.info(f"Completed translation to {lang}")
                except Exception as e:
                    self.logger.error(f"Failed translation to {lang}: {e}")
                    results[lang] = {"error": str(e)}
            
            return {
                "original_text": text,
                "translations": results,
                "total_languages": len(target_languages),
                "successful_translations": len([r for r in results.values() if "error" not in r])
            }
            
        except Exception as e:
            self.logger.error(f"Error in multi-language translation: {e}")
            raise

    def translation_confidence_analysis(self, text: str, target_lang: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze translation confidence by comparing multiple model outputs
        
        Args:
            text: Text to translate
            target_lang: Target language
            params: Generation parameters
            
        Returns:
            Confidence analysis results
        """
        try:
            # Get translations from all models
            results = self.translate(text, target_lang, params)
            successful_results = [r for r in results if r.get("output") is not None]
            
            if len(successful_results) < 2:
                return {
                    "confidence": "Low",
                    "reason": "Only one model available for comparison",
                    "translations": results
                }
            
            # Simple similarity analysis
            translations = [r["output"] for r in successful_results]
            
            # Calculate pairwise similarities (simple word overlap)
            similarities = []
            for i in range(len(translations)):
                for j in range(i + 1, len(translations)):
                    words1 = set(translations[i].lower().split())
                    words2 = set(translations[j].lower().split())
                    
                    if len(words1) == 0 or len(words2) == 0:
                        similarity = 0.0
                    else:
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        similarity = intersection / union if union > 0 else 0.0
                    
                    similarities.append(similarity)
            
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Determine confidence level
            if avg_similarity >= 0.7:
                confidence = "High"
                reason = "Models show high agreement"
            elif avg_similarity >= 0.4:
                confidence = "Medium"
                reason = "Models show moderate agreement"
            else:
                confidence = "Low"
                reason = "Models show low agreement"
            
            return {
                "confidence": confidence,
                "reason": reason,
                "average_similarity": avg_similarity,
                "individual_similarities": similarities,
                "translations": results,
                "recommendation": self._get_confidence_recommendation(confidence, avg_similarity)
            }
            
        except Exception as e:
            self.logger.error(f"Error in confidence analysis: {e}")
            raise

    def _get_confidence_recommendation(self, confidence: str, similarity: float) -> str:
        """Get recommendation based on confidence analysis"""
        if confidence == "High":
            return "Translation is likely accurate. Models are in strong agreement."
        elif confidence == "Medium":
            return "Translation appears reasonable but consider reviewing for accuracy."
        else:
            return "Translation may need review. Consider using additional context or a different approach."

    def get_translation_statistics(self) -> Dict[str, Any]:
        """Get translation usage statistics"""
        try:
            stats = {
                "available_models": len(self.models),
                "supported_languages": len(self.SUPPORTED_LANGUAGES),
                "language_list": list(self.SUPPORTED_LANGUAGES.values())
            }
            
            # Add model-specific stats if available
            for model in self.models:
                if hasattr(model, 'get_usage_stats'):
                    model_stats = model.get_usage_stats()
                    stats[f"{model.get_name()}_stats"] = model_stats
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting translation statistics: {e}")
            return {"error": str(e)}

    def benchmark_translation(self, test_texts: List[str], target_lang: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Benchmark translation performance across models
        
        Args:
            test_texts: List of test texts
            target_lang: Target language
            params: Generation parameters
            
        Returns:
            Benchmark results
        """
        try:
            benchmark_results = {
                "total_texts": len(test_texts),
                "target_language": target_lang,
                "model_performance": {},
                "overall_stats": {}
            }
            
            for model in self.models:
                model_name = model.get_name()
                model_results = {
                    "successful_translations": 0,
                    "failed_translations": 0,
                    "total_time": 0.0,
                    "average_time": 0.0,
                    "total_words": 0,
                    "errors": []
                }
                
                for i, text in enumerate(test_texts):
                    try:
                        result = self._translate_with_model(model, text, target_lang, params)
                        
                        if result.get("success"):
                            model_results["successful_translations"] += 1
                            model_results["total_time"] += result["inference_time"]
                            model_results["total_words"] += result["word_count"]
                        else:
                            model_results["failed_translations"] += 1
                            model_results["errors"].append(f"Text {i+1}: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        model_results["failed_translations"] += 1
                        model_results["errors"].append(f"Text {i+1}: {str(e)}")
                
                # Calculate averages
                if model_results["successful_translations"] > 0:
                    model_results["average_time"] = model_results["total_time"] / model_results["successful_translations"]
                    model_results["average_words"] = model_results["total_words"] / model_results["successful_translations"]
                
                benchmark_results["model_performance"][model_name] = model_results
            
            # Calculate overall statistics
            total_successful = sum(perf["successful_translations"] for perf in benchmark_results["model_performance"].values())
            total_failed = sum(perf["failed_translations"] for perf in benchmark_results["model_performance"].values())
            
            benchmark_results["overall_stats"] = {
                "total_attempts": total_successful + total_failed,
                "success_rate": total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0.0,
                "failure_rate": total_failed / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0.0
            }
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Error in translation benchmark: {e}")
            raise