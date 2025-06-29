from src.models.base_model import BaseModel
from utils.logger import setup_logger
from utils.helpers import validate_text
from typing import List, Dict, Any

class Chatter:
    def __init__(self, models: List[BaseModel]):
        self.logger = setup_logger(__name__)
        self.models = models
        self.logger.info("Chatter initialized with models: %s", [m.get_name() for m in self.models])

    def chat(self, messages: list, params: Dict[str, Any]) -> List[Dict]:
        if not messages or not messages[-1]["content"]:
            raise ValueError("Chat input cannot be empty")
        results = []
        for model in self.models:
            try:
                response, inference_time = model.chat(messages, params)
                results.append({
                    "model": model.get_name(),
                    "output": response,
                    "inference_time": inference_time
                })
            except Exception as e:
                self.logger.error(f"Error chatting with {model.get_name()}: {str(e)}")
                results.append({
                    "model": model.get_name(),
                    "output": None,
                    "inference_time": None,
                    "error": str(e)
                })
        return results