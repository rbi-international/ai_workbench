from src.models.base_model import BaseModel
from utils.logger import setup_logger
from utils.helpers import validate_text
from typing import List, Dict, Any

class Translator:
    def __init__(self, models: List[BaseModel]):
        self.logger = setup_logger(__name__)
        self.models = models
        self.logger.info("Translator initialized with models: %s", [m.get_name() for m in self.models])

    def translate(self, text: str, target_lang: str, params: Dict[str, Any]) -> List[Dict]:
        text = validate_text(text)
        results = []
        for model in self.models:
            try:
                translation, inference_time = model.translate(text, target_lang, params)
                results.append({
                    "model": model.get_name(),
                    "output": translation,
                    "inference_time": inference_time
                })
            except Exception as e:
                self.logger.error(f"Error translating with {model.get_name()}: {str(e)}")
                results.append({
                    "model": model.get_name(),
                    "output": None,
                    "inference_time": None,
                    "error": str(e)
                })
        return results