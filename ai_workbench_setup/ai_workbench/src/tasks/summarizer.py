from src.models.base_model import BaseModel
from utils.logger import setup_logger
from utils.helpers import validate_text, validate_model_params, clean_response_text
from typing import List, Dict, Any, Optional
import time

class Summarizer:
    """
    Summarizer class that handles text summarization across multiple models
    """
    
    def __init__(self, models: List[BaseModel]):
        self.logger = setup_logger(__name__)
        
        # Filter only available models
        self.models = [model for model in models if model.is_available()]
        
        if not self.models:
            self.logger.warning("No available models for summarization")
        else:
            model_names = [model.get_name() for model in self.models]
            self.logger.info(f"Summarizer initialized with models: {model_names}")

    def summarize(self, text: str, params: Dict[str, Any]) -> List[Dict]:
        """
        Generate summaries using all available models
        
        Args:
            text: Text to summarize
            params: Generation parameters
            
        Returns:
            List of result dictionaries with model outputs
        """
        try:
            # Validate inputs
            text = validate_text(text)
            params = validate_model_params(params)
            
            self.logger.info(f"Starting summarization for {len(self.models)} models")
            self.logger.debug(f"Input text length: {len(text)} characters")
            self.logger.debug(f"Parameters: {params}")
            
            results = []
            
            for model in self.models:
                model_result = self._summarize_with_model(model, text, params)
                results.append(model_result)
            
            # Log summary statistics
            successful_results = [r for r in results if r.get("output") is not None]
            self.logger.info(
                f"Summarization completed: {len(successful_results)}/{len(self.models)} models succeeded"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in summarization: {e}")
            raise

    def _summarize_with_model(self, model: BaseModel, text: str, params: Dict[str, Any]) -> Dict:
        """
        Generate summary with a single model
        
        Args:
            model: Model instance
            text: Text to summarize
            params: Generation parameters
            
        Returns:
            Result dictionary
        """
        model_name = model.get_name()
        start_time = time.time()
        
        try:
            self.logger.debug(f"Starting summarization with {model_name}")
            
            # Check if model is available
            if not model.is_available():
                raise RuntimeError(f"Model {model_name} is not available")
            
            # Check input length
            if not model.check_input_length(text):
                self.logger.warning(f"Input text may be too long for {model_name}")
            
            # Generate summary
            summary, inference_time = model.summarize(text, params)
            
            # Validate output
            if not summary or not isinstance(summary, str):
                raise ValueError("Model returned empty or invalid summary")
            
            result = {
                "model": model_name,
                "output": clean_response_text(summary.strip()),
                "inference_time": inference_time,
                "success": True,
                "word_count": len(summary.split()),
                "character_count": len(summary)
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
                "error_type": type(e).__name__
            }

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return [model.get_name() for model in self.models if model.is_available()]

    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about all models"""
        return [model.get_model_info() for model in self.models]

    def summarize_with_specific_model(self, text: str, model_name: str, params: Dict[str, Any]) -> Dict:
        """
        Generate summary with a specific model
        
        Args:
            text: Text to summarize
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
            params = validate_model_params(params)
            
            # Generate summary
            return self._summarize_with_model(target_model, text, params)
            
        except Exception as e:
            self.logger.error(f"Error in model-specific summarization: {e}")
            raise

    def batch_summarize(self, texts: List[str], params: Dict[str, Any]) -> List[List[Dict]]:
        """
        Summarize multiple texts
        
        Args:
            texts: List of texts to summarize
            params: Generation parameters
            
        Returns:
            List of result lists (one per input text)
        """
        try:
            self.logger.info(f"Starting batch summarization for {len(texts)} texts")
            
            results = []
            for i, text in enumerate(texts):
                self.logger.debug(f"Processing text {i+1}/{len(texts)}")
                text_results = self.summarize(text, params)
                results.append(text_results)
            
            self.logger.info("Batch summarization completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch summarization: {e}")
            raise

    def compare_summaries(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summaries and provide comparison metrics
        
        Args:
            text: Text to summarize
            params: Generation parameters
            
        Returns:
            Comparison results
        """
        try:
            # Generate summaries
            results = self.summarize(text, params)
            
            # Calculate comparison metrics
            successful_results = [r for r in results if r.get("output") is not None]
            
            if not successful_results:
                return {"error": "No successful summaries generated"}
            
            # Basic statistics
            word_counts = [r["word_count"] for r in successful_results]
            inference_times = [r["inference_time"] for r in successful_results]
            
            comparison = {
                "summaries": results,
                "statistics": {
                    "total_models": len(results),
                    "successful_models": len(successful_results),
                    "average_word_count": sum(word_counts) / len(word_counts),
                    "average_inference_time": sum(inference_times) / len(inference_times),
                    "fastest_model": min(successful_results, key=lambda x: x["inference_time"])["model"],
                    "shortest_summary": min(successful_results, key=lambda x: x["word_count"])["model"],
                    "longest_summary": max(successful_results, key=lambda x: x["word_count"])["model"]
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error in summary comparison: {e}")
            raise

    def validate_summary_quality(self, summary: str, original_text: str) -> Dict[str, Any]:
        """
        Basic validation of summary quality
        
        Args:
            summary: Generated summary
            original_text: Original text
            
        Returns:
            Quality metrics
        """
        try:
            if not summary or not original_text:
                return {"error": "Empty summary or original text"}
            
            # Basic metrics
            summary_words = len(summary.split())
            original_words = len(original_text.split())
            compression_ratio = summary_words / original_words if original_words > 0 else 0
            
            # Check for common issues
            issues = []
            if compression_ratio > 0.8:
                issues.append("Summary may be too long (low compression)")
            if compression_ratio < 0.05:
                issues.append("Summary may be too short (high compression)")
            if summary.lower() == original_text.lower():
                issues.append("Summary is identical to original")
            
            return {
                "summary_words": summary_words,
                "original_words": original_words,
                "compression_ratio": compression_ratio,
                "compression_percentage": f"{(1 - compression_ratio) * 100:.1f}%",
                "issues": issues,
                "quality_score": self._calculate_quality_score(compression_ratio, issues)
            }
            
        except Exception as e:
            self.logger.error(f"Error in summary validation: {e}")
            return {"error": str(e)}

    def _calculate_quality_score(self, compression_ratio: float, issues: List[str]) -> float:
        """Calculate a basic quality score"""
        base_score = 1.0
        
        # Penalize for compression issues
        if compression_ratio > 0.8:
            base_score -= 0.3
        elif compression_ratio < 0.05:
            base_score -= 0.4
        
        # Penalize for each issue
        base_score -= len(issues) * 0.2
        
        return max(0.0, min(1.0, base_score))