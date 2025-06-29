from utils.logger import setup_logger

class CostTracker:
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.usage = {}
        self.max_words = 100

    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int, output_words: int):
        if model not in self.usage:
            self.usage[model] = {"prompt_tokens": 0, "completion_tokens": 0, "output_words": 0}
        self.usage[model]["prompt_tokens"] += prompt_tokens
        self.usage[model]["completion_tokens"] += completion_tokens
        self.usage[model]["output_words"] += output_words
        if output_words > self.max_words:
            self.logger.warning(f"Output for {model} exceeds word limit: {output_words} words")
        self.logger.info(f"Usage for {model}: {self.usage[model]}")