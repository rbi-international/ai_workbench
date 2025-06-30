import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import re

def validate_text(text: str) -> str:
    """
    Validate and clean input text
    
    Args:
        text: Input text to validate
        
    Returns:
        Cleaned text
        
    Raises:
        ValueError: If text is invalid
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")
    
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Input text cannot be empty or only whitespace")
    
    return cleaned_text

def clean_response_text(text: str) -> str:
    """
    Clean response text by removing unwanted formatting while preserving readability
    """
    if not text:
        return text
    
    # Remove only problematic escaped characters, preserve actual newlines
    cleaned = text.replace('\\n', ' ')     # Only escaped \n
    cleaned = cleaned.replace('\\r', ' ')   # Only escaped \r
    cleaned = cleaned.replace('\\t', ' ')   # Only escaped \t
    
    # Clean up multiple spaces but preserve single spaces between words
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove excessive punctuation repetition
    cleaned = re.sub(r'([.!?])\1{2,}', r'\1', cleaned)
    
    # Clean up markdown formatting if needed
    cleaned = cleaned.replace('**', '')    # Bold markdown
    
    # Clean up bullet points and dashes only if they're causing issues
    cleaned = re.sub(r'\s*-\s*-\s*', ' - ', cleaned)  # Multiple dashes
    
    return cleaned.strip()

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def load_json_file(file_path: Union[str, Path]) -> Optional[Dict]:
    """
    Load JSON file with error handling
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary or None if file doesn't exist
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

def save_json_file(data: Dict, file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save file
    """
    # Ensure parent directory exists
    ensure_directory(Path(file_path).parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_cache_key(*args) -> str:
    """
    Generate cache key from arguments
    
    Args:
        *args: Arguments to hash
        
    Returns:
        Cache key string
    """
    # Convert all arguments to string and concatenate
    key_string = "_".join(str(arg) for arg in args)
    
    # Create hash for consistent key length
    return hashlib.md5(key_string.encode()).hexdigest()

def truncate_text(text: str, max_words: int) -> str:
    """
    Truncate text to maximum number of words
    
    Args:
        text: Text to truncate
        max_words: Maximum number of words
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    words = text.split()
    if len(words) <= max_words:
        return text
    
    return " ".join(words[:max_words]) + "..."

def safe_get_env(key: str, default: Any = None, required: bool = False) -> str:
    """
    Safely get environment variable
    
    Args:
        key: Environment variable key
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If required variable is missing
    """
    value = os.getenv(key, default)
    
    if required and not value:
        raise ValueError(f"Required environment variable {key} is not set")
    
    return value

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load configuration file with error handling
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")

def format_inference_time(seconds: float) -> str:
    """
    Format inference time for display
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def validate_model_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize model parameters
    
    Args:
        params: Model parameters dictionary
        
    Returns:
        Validated parameters
    """
    validated = {}
    
    # Temperature validation
    temp = params.get('temperature', 0.7)
    validated['temperature'] = max(0.1, min(2.0, float(temp)))
    
    # Top-p validation
    top_p = params.get('top_p', 0.9)
    validated['top_p'] = max(0.1, min(1.0, float(top_p)))
    
    # Token limits validation
    max_tokens = params.get('max_tokens', 100)
    validated['max_tokens'] = max(10, min(4000, int(max_tokens)))
    
    min_tokens = params.get('min_tokens', 30)
    validated['min_tokens'] = max(1, min(validated['max_tokens'], int(min_tokens)))
    
    return validated