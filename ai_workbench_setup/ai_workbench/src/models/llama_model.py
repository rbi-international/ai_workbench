from src.models.base_model import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from dotenv import load_dotenv
from utils.logger import setup_logger
from utils.cost_tracker import CostTracker
from utils.helpers import validate_text, ensure_directory, generate_cache_key, truncate_text, clean_response_text

load_dotenv()

class LlamaModel(BaseModel):
    """
    LLaMA model implementation with local inference
    """
    
    def __init__(self, config: dict):
        self.logger = setup_logger(__name__)
        self.cost_tracker = CostTracker()
        
        # Model configuration
        self.model_name = config.get("name", "meta-llama/Llama-3.1-8B-Instruct")
        self.max_output_words = config.get("max_output_words", 100)
        self.enabled = config.get("enabled", False)
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Setup cache directory
        self.cache_dir = Path(config.get("cache", {}).get("path", "data/cache/"))
        ensure_directory(self.cache_dir)
        
        # Initialize model and tokenizer only if enabled
        self.tokenizer = None
        self.model = None
        
        if self.enabled:
            self._initialize_model()
        else:
            self.logger.info(f"LLaMA model {self.model_name} is disabled")

    def _initialize_model(self):
        """Initialize the LLaMA model and tokenizer"""
        try:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            
            self.logger.info(f"Loading LLaMA model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                token=hf_token,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations for CPU/GPU
            model_kwargs = {
                "token": hf_token,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            # For CPU inference, use smaller precision
            if self.device.type == "cpu":
                model_kwargs["torch_dtype"] = torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            ).to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.logger.info(f"✓ LLaMA model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLaMA model: {e}")
            self.enabled = False
            raise

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

    def _generate_response(self, prompt: str, params: Dict[str, Any]) -> Tuple[str, float]:
        """
        Generate response using the LLaMA model
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            
        Returns:
            Tuple of (response_text, inference_time)
        """
        if not self.enabled or not self.model or not self.tokenizer:
            raise RuntimeError("LLaMA model is not enabled or not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048,
                padding=True
            ).to(self.device)
            
            start_time = time.time()
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=params.get("max_tokens", 100),
                    min_new_tokens=params.get("min_tokens", 30),
                    temperature=params.get("temperature", 0.7),
                    top_p=params.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            inference_time = time.time() - start_time
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Clean up response
            response = clean_response_text(response.strip())
            
            return response, inference_time
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    def summarize(self, text: str, params: Dict[str, Any]) -> Tuple[str, float]:
        """Generate summary using LLaMA model"""
        try:
            if not self.enabled:
                raise RuntimeError("LLaMA model is disabled")
            
            # Validate input
            text = validate_text(text)
            params = self.validate_params(params)
            
            # Generate cache key
            cache_key = generate_cache_key(
                "summarize", 
                text, 
                self.model_name,
                params["max_tokens"],
                params["temperature"],
                params["top_p"]
            )
            
            # Check cache
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached summary for {self.model_name}")
                return cached["summary"], cached["inference_time"]
            
            # Create prompt
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that creates concise, accurate summaries.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Please summarize the following text in approximately {params['min_tokens']}-{params['max_tokens']} words:

{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            
            # Generate response
            response, inference_time = self._generate_response(prompt, params)
            
            # Truncate output if needed
            summary = truncate_text(response, self.max_output_words)
            
            # Track usage (estimate tokens for local model)
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(summary)
            
            self.cost_tracker.log_usage(
                self.model_name,
                input_tokens,
                output_tokens,
                len(summary.split())
            )
            
            # Cache result
            cache_data = {
                "summary": summary,
                "inference_time": inference_time
            }
            self._save_cache(cache_key, cache_data)
            
            self.logger.info(f"Generated summary in {inference_time:.2f}s")
            return summary, inference_time
            
        except Exception as e:
            self.logger.error(f"Error in summarization: {e}")
            raise

    def translate(self, text: str, target_lang: str, params: Dict[str, Any]) -> Tuple[str, float]:
        """Translate text using LLaMA model"""
        try:
            if not self.enabled:
                raise RuntimeError("LLaMA model is disabled")
            
            # Validate input
            text = validate_text(text)
            params = self.validate_params(params)
            
            if not target_lang:
                raise ValueError("Target language cannot be empty")
            
            # Generate cache key
            cache_key = generate_cache_key(
                "translate",
                text,
                target_lang,
                self.model_name,
                params["max_tokens"],
                params["temperature"],
                params["top_p"]
            )
            
            # Check cache
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached translation for {self.model_name}")
                return cached["translation"], cached["inference_time"]
            
            # Create prompt
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional translator. Translate text accurately while maintaining the original meaning and tone.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate the following text to {target_lang}:

{text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            
            # Generate response
            response, inference_time = self._generate_response(prompt, params)
            
            # Truncate output if needed
            translation = truncate_text(response, self.max_output_words)
            
            # Track usage
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(translation)
            
            self.cost_tracker.log_usage(
                self.model_name,
                input_tokens,
                output_tokens,
                len(translation.split())
            )
            
            # Cache result
            cache_data = {
                "translation": translation,
                "inference_time": inference_time
            }
            self._save_cache(cache_key, cache_data)
            
            self.logger.info(f"Generated translation in {inference_time:.2f}s")
            return translation, inference_time
            
        except Exception as e:
            self.logger.error(f"Error in translation: {e}")
            raise

    def chat(self, messages: List[Dict], params: Dict[str, Any]) -> Tuple[str, float]:
        """Generate chat response using LLaMA model"""
        try:
            if not self.enabled:
                raise RuntimeError("LLaMA model is disabled")
            
            # Validate input
            if not messages or not messages[-1].get("content"):
                raise ValueError("Chat input cannot be empty")
            
            params = self.validate_params(params)
            messages = self.format_messages(messages)
            
            # Generate cache key
            cache_key = generate_cache_key(
                "chat",
                str(messages),
                self.model_name,
                params["max_tokens"],
                params["temperature"],
                params["top_p"]
            )
            
            # Check cache
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached chat response for {self.model_name}")
                return cached["response"], cached["inference_time"]
            
            # Create conversation prompt
            prompt = "<|begin_of_text|>"
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt += f"<|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>"
                elif role == "user":
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
                elif role == "assistant":
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"
            
            # Add assistant start token
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            
            # Generate response
            response, inference_time = self._generate_response(prompt, params)
            
            # Truncate output if needed
            response = truncate_text(response, self.max_output_words)
            
            # Track usage
            input_tokens = self.estimate_tokens(prompt)
            output_tokens = self.estimate_tokens(response)
            
            self.cost_tracker.log_usage(
                self.model_name,
                input_tokens,
                output_tokens,
                len(response.split())
            )
            
            # Cache result
            cache_data = {
                "response": response,
                "inference_time": inference_time
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
        return self.enabled and self.model is not None and self.tokenizer is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            "device": str(self.device),
            "enabled": self.enabled,
            "loaded": self.model is not None,
            "memory_usage": self._get_memory_usage()
        })
        return info

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                return {
                    "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                    "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
                    "total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                }
            else:
                import psutil
                return {
                    "ram_usage": f"{psutil.virtual_memory().percent}%",
                    "available": f"{psutil.virtual_memory().available / 1024**3:.2f} GB"
                }
        except Exception:
            return {"error": "Unable to get memory info"}

    def cleanup(self):
        """Clean up model resources"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            
            self.logger.info("Model resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to clean up resources"""
        try:
            self.cleanup()
        except Exception:
            pass