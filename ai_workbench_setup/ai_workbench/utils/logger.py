import logging
import yaml
import os
from pathlib import Path
from typing import Optional

def setup_logger(name: str, config_path: str = "config/config.yaml") -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name (usually __name__)
        config_path: Path to configuration file
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Load configuration with fallback
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        log_level = config.get('logging', {}).get('level', 'INFO')
        log_format = config.get('logging', {}).get('format', 
                               "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    except (FileNotFoundError, yaml.YAMLError):
        # Fallback configuration
        log_level = 'INFO'
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Set level
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # File handler for general logs
    try:
        file_handler = logging.FileHandler(log_dir / "ai_workbench.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except PermissionError:
        # If we can't write to file, just use console
        pass
    
    # Cost tracking file handler
    try:
        cost_handler = logging.FileHandler(log_dir / "cost_tracker.log", encoding='utf-8')
        cost_handler.setFormatter(formatter)
        cost_handler.setLevel(logging.INFO)
        
        # Only add cost handler for cost tracker
        if 'cost' in name.lower():
            logger.addHandler(cost_handler)
    except PermissionError:
        pass
    
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name)