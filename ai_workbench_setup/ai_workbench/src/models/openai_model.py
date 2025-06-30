from src.models.base_model import BaseModel
from openai import OpenAI
import time
import hashlib
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
from dotenv import load_dotenv
from utils.logger import setup_logger
from utils.cost_tracker import CostTracker
from utils.helpers import validate_text, ensure_directory, generate_cache_key, truncate_text, clean_response_text

load_dotenv()

class OpenAIModel(BaseModel):
    """
    OpenAI model implementation with caching, error handling, and cost tracking
    """
    
    def __init__(self, config: dict):
        self.logger = setup_logger(__name__)
        self.cost_tracker = CostTracker()
        
        # Model configuration
        self.model_name = config.get("name", "gpt-4o")
        self.max_output_words = config.get("max_output_words", 100)
        self.enabled = config.get("enabled", True)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OpenAI API key is required but not found in environment")
        
        try:
            self.client = OpenAI(api_key=api_key)
            self.logger.info(f"OpenAI client initialized for model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Setup cache directory
        self.cache_dir = Path(config.get("cache", {}).get("path", "data/cache/"))
        ensure_directory(self.cache_dir)
        
        # Test API connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test OpenAI API connection"""
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            self.logger.info("✓ OpenAI API connection successful")
            return True
        except Exception as e:
            self.logger.warning(f"OpenAI API connection test failed: {e}")
            return False

    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load cached response"""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.debug(f"Cache hit for key: {key}")
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.debug(f"Cache miss or error for key {key}: {e}")
        return None

    def _save_cache(self, key: str, data: Dict):
        """Save response to cache"""
        try:
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Cached response for key: {key}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache for key {key}: {e}")

    def _make_api_call(self, messages: List[Dict], params: Dict[str, Any]) -> Tuple[str, float, Dict]:
        """
        Make API call to OpenAI with error handling and retries
        
        Args:
            messages: List of message dictionaries
            params: Generation parameters
            
        Returns:
            Tuple of (response_text, inference_time, usage_data)
        """
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=params.get("max_tokens", 100),
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 0.9),
                    presence_penalty=0,
                    frequency_penalty=0
                )
                
                inference_time = time.time() - start_time
                
                # Extract response
                response_text = response.choices[0].message.content
                if not response_text:
                    response_text = ""
                
                # Extract usage data
                usage_data = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                
                return response_text, inference_time, usage_data
                
            except Exception as e:
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All API call attempts failed for {self.model_name}")
                    raise

    def summarize(self, text: str, params: Dict[str, Any]) -> Tuple[str, float]:
        """
        Generate summary using OpenAI API
        
        Args:
            text: Text to summarize
            params: Generation parameters
            
        Returns:
            Tuple of (summary, inference_time)
        """
        try:
            # Validate input
            text = validate_text(text)
            
            # Generate cache key
            cache_key = generate_cache_key(
                "summarize", 
                text, 
                self.model_name,
                params.get("max_tokens", 100),
                params.get("temperature", 0.7),
                params.get("top_p", 0.9)
            )
            
            # Check cache
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached summary for {self.model_name}")
                return cached["summary"], cached["inference_time"]
            
            # Prepare messages
            system_prompt = (
                "You are a helpful assistant that creates concise, accurate summaries. "
                f"Provide a summary in approximately {params.get('min_tokens', 30)}-"
                f"{params.get('max_tokens', 100)} words."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
            ]
            
            # Make API call
            response_text, inference_time, usage_data = self._make_api_call(messages, params)
            
            # Truncate output if needed
            summary = truncate_text(response_text, self.max_output_words)
            
            # Track usage
            self.cost_tracker.log_usage(
                self.model_name,
                usage_data["prompt_tokens"],
                usage_data["completion_tokens"],
                len(summary.split())
            )
            
            # Cache result
            cache_data = {
                "summary": summary,
                "inference_time": inference_time,
                "usage": usage_data
            }
            self._save_cache(cache_key, cache_data)
            
            self.logger.info(f"Generated summary in {inference_time:.2f}s")
            return summary, inference_time
            
        except Exception as e:
            self.logger.error(f"Error in summarization: {e}")
            raise

    def translate(self, text: str, target_lang: str, params: Dict[str, Any]) -> Tuple[str, float]:
        """
        Translate text using OpenAI API
        
        Args:
            text: Text to translate
            target_lang: Target language
            params: Generation parameters
            
        Returns:
            Tuple of (translation, inference_time)
        """
        try:
            # Validate input
            text = validate_text(text)
            if not target_lang:
                raise ValueError("Target language cannot be empty")
            
            # Generate cache key
            cache_key = generate_cache_key(
                "translate",
                text,
                target_lang,
                self.model_name,
                params.get("max_tokens", 100),
                params.get("temperature", 0.7),
                params.get("top_p", 0.9)
            )
            
            # Check cache
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached translation for {self.model_name}")
                return cached["translation"], cached["inference_time"]
            
            # Prepare messages
            system_prompt = (
                f"You are a professional translator. Translate the following text to {target_lang}. "
                "Provide only the translation, maintaining the original meaning and tone."
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            
            # Make API call
            response_text, inference_time, usage_data = self._make_api_call(messages, params)
            
            # Truncate output if needed
            translation = truncate_text(response_text, self.max_output_words)
            
            # Track usage
            self.cost_tracker.log_usage(
                self.model_name,
                usage_data["prompt_tokens"],
                usage_data["completion_tokens"],
                len(translation.split())
            )
            
            # Cache result
            cache_data = {
                "translation": translation,
                "inference_time": inference_time,
                "usage": usage_data
            }
            self._save_cache(cache_key, cache_data)
            
            self.logger.info(f"Generated translation in {inference_time:.2f}s")
            return translation, inference_time
            
        except Exception as e:
            self.logger.error(f"Error in translation: {e}")
            raise

    def chat(self, messages: List[Dict], params: Dict[str, Any]) -> Tuple[str, float]:
        """
        Generate chat response using OpenAI API
        
        Args:
            messages: List of conversation messages
            params: Generation parameters
            
        Returns:
            Tuple of (response, inference_time)
        """
        try:
            # Validate input
            if not messages or not messages[-1].get("content"):
                raise ValueError("Chat input cannot be empty")
            
            # Generate cache key
            cache_key = generate_cache_key(
                "chat",
                str(messages),
                self.model_name,
                params.get("max_tokens", 100),
                params.get("temperature", 0.7),
                params.get("top_p", 0.9)
            )
            
            # Check cache
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached chat response for {self.model_name}")
                return cached["response"], cached["inference_time"]
            
            # Add system message if not present
            if not messages or messages[0].get("role") != "system":
                system_message = {
                    "role": "system", 
                    "content": "You are a helpful, knowledgeable, and friendly AI assistant."
                }
                messages = [system_message] + messages
            
            # Make API call
            response_text, inference_time, usage_data = self._make_api_call(messages, params)
            
            # Truncate output if needed
            response = truncate_text(response_text, self.max_output_words)
            
            # Track usage
            self.cost_tracker.log_usage(
                self.model_name,
                usage_data["prompt_tokens"],
                usage_data["completion_tokens"],
                len(response.split())
            )
            
            # Cache result
            cache_data = {
                "response": response,
                "inference_time": inference_time,
                "usage": usage_data
            }
            self._save_cache(cache_key, cache_data)
            
            self.logger.info(f"Generated chat response in {inference_time:.2f}s")
            return response, inference_time
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            raise

    def get_name(self) -> str:
        """Get model name"""
        return self.model_name

    def is_available(self) -> bool:
        """Check if model is available"""
        return self.enabled and self._test_connection()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this model"""
        try:
            summary = self.cost_tracker.get_usage_summary()
            model_stats = summary.get("models", {}).get(self.model_name, {})
            return {
                "model": self.model_name,
                "total_requests": model_stats.get("requests", 0),
                "total_tokens": model_stats.get("total_tokens", 0),
                "total_cost": model_stats.get("cost", 0.0),
                "average_cost_per_request": (
                    model_stats.get("cost", 0.0) / max(model_stats.get("requests", 1), 1)
                )
            }
        except Exception as e:
            self.logger.error(f"Error getting usage stats: {e}")
            return {"error": str(e)}