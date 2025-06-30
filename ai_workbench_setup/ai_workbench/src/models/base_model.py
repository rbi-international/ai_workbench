from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional

class BaseModel(ABC):
    """
    Abstract base class for all AI models
    
    This defines the interface that all model implementations must follow
    """
    
    @abstractmethod
    def summarize(self, text: str, params: Dict[str, Any]) -> Tuple[str, float]:
        """
        Generate a summary of the input text
        
        Args:
            text: Text to summarize
            params: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Tuple of (summary_text, inference_time_seconds)
        """
        pass

    @abstractmethod
    def translate(self, text: str, target_lang: str, params: Dict[str, Any]) -> Tuple[str, float]:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_lang: Target language (e.g., "Spanish", "French")
            params: Generation parameters
            
        Returns:
            Tuple of (translated_text, inference_time_seconds)
        """
        pass

    @abstractmethod
    def chat(self, messages: List[Dict], params: Dict[str, Any]) -> Tuple[str, float]:
        """
        Generate chat response
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            params: Generation parameters
            
        Returns:
            Tuple of (response_text, inference_time_seconds)
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the model name/identifier
        
        Returns:
            Model name string
        """
        pass

    def truncate_output(self, text: str, max_words: int) -> str:
        """
        Truncate text output to maximum number of words
        
        Args:
            text: Text to truncate
            max_words: Maximum number of words allowed
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        words = text.split()
        if len(words) <= max_words:
            return text
        
        # Truncate and add ellipsis
        truncated = " ".join(words[:max_words])
        return truncated + "..."

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and sanitize generation parameters
        
        Args:
            params: Input parameters
            
        Returns:
            Validated parameters
        """
        validated = {}
        
        # Temperature validation (0.1 to 2.0)
        temp = params.get('temperature', 0.7)
        try:
            temp = float(temp)
            validated['temperature'] = max(0.1, min(2.0, temp))
        except (ValueError, TypeError):
            validated['temperature'] = 0.7
        
        # Top-p validation (0.1 to 1.0)
        top_p = params.get('top_p', 0.9)
        try:
            top_p = float(top_p)
            validated['top_p'] = max(0.1, min(1.0, top_p))
        except (ValueError, TypeError):
            validated['top_p'] = 0.9
        
        # Max tokens validation
        max_tokens = params.get('max_tokens', 100)
        try:
            max_tokens = int(max_tokens)
            validated['max_tokens'] = max(10, min(4000, max_tokens))
        except (ValueError, TypeError):
            validated['max_tokens'] = 100
        
        # Min tokens validation
        min_tokens = params.get('min_tokens', 30)
        try:
            min_tokens = int(min_tokens)
            validated['min_tokens'] = max(1, min(validated['max_tokens'], min_tokens))
        except (ValueError, TypeError):
            validated['min_tokens'] = 30
        
        return validated

    def is_available(self) -> bool:
        """
        Check if the model is available for use
        
        Returns:
            True if model is available, False otherwise
        """
        return True  # Default implementation

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        
        Returns:
            Dictionary with model information
        """
        return {
            "name": self.get_name(),
            "type": self.__class__.__name__,
            "available": self.is_available()
        }

    def format_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Format and validate chat messages
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted and validated messages
        """
        if not messages:
            return []
        
        formatted = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            # Validate role
            if role not in ['system', 'user', 'assistant']:
                role = 'user'
            
            # Ensure content is string
            if not isinstance(content, str):
                content = str(content)
            
            # Skip empty messages
            if not content.strip():
                continue
            
            formatted.append({
                'role': role,
                'content': content.strip()
            })
        
        return formatted

    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count for text
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        
        # Rough estimation: ~4 characters per token for English
        return len(text) // 4

    def check_input_length(self, text: str, max_length: int = 8000) -> bool:
        """
        Check if input text is within acceptable length
        
        Args:
            text: Input text
            max_length: Maximum allowed length in characters
            
        Returns:
            True if within limits, False otherwise
        """
        return len(text) <= max_length if text else True