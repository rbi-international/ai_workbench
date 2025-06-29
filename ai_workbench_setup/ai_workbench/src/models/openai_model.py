from src.models.base_model import BaseModel
from openai import OpenAI
import time
from utils.logger import setup_logger
from utils.cost_tracker import CostTracker
from typing import Tuple, Dict, Any
import os
from dotenv import load_dotenv
import json

load_dotenv()

class OpenAIModel(BaseModel):
    def __init__(self, config: dict):
        self.logger = setup_logger(__name__)
        self.cost_tracker = CostTracker()
        self.model_name = config["name"]
        self.max_output_words = config["max_output_words"]
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache_dir = config.get("cache", {}).get("path", "data/cache/")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Initialized {self.model_name}")

    def _load_cache(self, key: str) -> Dict:
        try:
            with open(f"{self.cache_dir}/{key}.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def _save_cache(self, key: str, data: Dict):
        with open(f"{self.cache_dir}/{key}.json", "w") as f:
            json.dump(data, f)

    def summarize(self, text: str, params: Dict[str, Any]) -> Tuple[str, float]:
        try:
            cache_key = f"summarize_{hash(text)}_{params['max_tokens']}_{params['temperature']}_{params['top_p']}"
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached summary for {self.model_name}")
                return cached["summary"], cached["inference_time"]

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                    {"role": "user", "content": f"Summarize in {params['min_tokens']}-{params['max_tokens']} tokens: {text}"}
                ],
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"]
            )
            summary = self.truncate_output(response.choices[0].message.content, self.max_output_words)
            inference_time = time.time() - start_time
            self.cost_tracker.log_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens, len(summary.split()))
            self._save_cache(cache_key, {"summary": summary, "inference_time": inference_time})
            self.logger.info(f"Generated summary in {inference_time:.2f}s")
            return summary, inference_time
        except Exception as e:
            self.logger.error(f"Error in summarization: {str(e)}")
            raise

    def translate(self, text: str, target_lang: str, params: Dict[str, Any]) -> Tuple[str, float]:
        try:
            cache_key = f"translate_{hash(text)}_{target_lang}_{params['max_tokens']}_{params['temperature']}_{params['top_p']}"
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached translation for {self.model_name}")
                return cached["translation"], cached["inference_time"]

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You are a translator to {target_lang}."},
                    {"role": "user", "content": f"Translate: {text}"}
                ],
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"]
            )
            translation = self.truncate_output(response.choices[0].message.content, self.max_output_words)
            inference_time = time.time() - start_time
            self.cost_tracker.log_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens, len(translation.split()))
            self._save_cache(cache_key, {"translation": translation, "inference_time": inference_time})
            self.logger.info(f"Generated translation in {inference_time:.2f}s")
            return translation, inference_time
        except Exception as e:
            self.logger.error(f"Error in translation: {str(e)}")
            raise

    def chat(self, messages: list, params: Dict[str, Any]) -> Tuple[str, float]:
        try:
            cache_key = f"chat_{hash(str(messages))}_{params['max_tokens']}_{params['temperature']}_{params['top_p']}"
            cached = self._load_cache(cache_key)
            if cached:
                self.logger.info(f"Using cached chat response for {self.model_name}")
                return cached["response"], cached["inference_time"]

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"]
            )
            response_text = self.truncate_output(response.choices[0].message.content, self.max_output_words)
            inference_time = time.time() - start_time
            self.cost_tracker.log_usage(self.model_name, response.usage.prompt_tokens, response.usage.completion_tokens, len(response_text.split()))
            self._save_cache(cache_key, {"response": response_text, "inference_time": inference_time})
            self.logger.info(f"Generated chat response in {inference_time:.2f}s")
            return response_text, inference_time
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            raise

    def get_name(self) -> str:
        return self.model_name