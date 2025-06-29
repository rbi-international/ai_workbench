import logging
import yaml
import os

def setup_logger(name: str) -> logging.Logger:
    with open("config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    logger = logging.getLogger(name)
    logger.setLevel(config['logging']['level'])
    
    # Console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(config['logging']['format'])
    console_handler.setFormatter(formatter)
    
    # File handler
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler("logs/cost_tracker.log")
    file_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger