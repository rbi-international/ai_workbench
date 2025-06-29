from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class BaseModel(ABC):
    @abstractmethod
    def summarize(self, text: str, params: Dict[str, Any]) -> Tuple[str, float]:
        pass

    @abstractmethod
    def translate(self, text: str, target_lang: str, params: Dict[str, Any]) -> Tuple[str, float]:
        pass

    @abstractmethod
    def chat(self, messages: list, params: Dict[str, Any]) -> Tuple[str, float]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def truncate_output(self, text: str, max_words: int) -> str:
        words = text.split()
        return " ".join(words[:max_words]) if len(words) > max_words else text