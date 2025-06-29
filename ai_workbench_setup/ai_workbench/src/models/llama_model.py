from src.models.base_model import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from utils.logger import setup_logger
from utils.cost_tracker import CostTracker
from typing import Tuple, Dict, Any
import os
from dotenv import load_dotenv
import json

load_dotenv()

class LlamaModel(BaseModel):
    def __init__(self, config: dict):
        self.logger = setup_logger(__name__)
        self.cost_tracker = CostTracker()
        self.model_name = config["name"]
        self.max_output_words = config["max_output_words"]
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=os.getenv("HUGGINGFACE_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=os.getenv("HUGGINGFACE_TOKEN")).to(self.device)
        self.cache_dir = config.get("cache", {}).get("path", "data/cache/")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"Loading {self.model_name} on {self.device}")

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

            prompt = f"Summarize in {params['min_tokens']}-{params['max_tokens']} tokens: {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                max_length=params["max_tokens"],
                min_length=params["min_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                do_sample=True
            )
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = self.truncate_output(summary, self.max_output_words)
            inference_time = time.time() - start_time
            self.cost_tracker.log_usage(self.model_name, len(inputs["input_ids"][0]), len(outputs[0]), len(summary.split()))
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

            prompt = f"Translate to {target_lang}: {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                max_length=params["max_tokens"],
                min_length=params["min_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                do_sample=True
            )
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translation = self.truncate_output(translation, self.max_output_words)
            inference_time = time.time() - start_time
            self.cost_tracker.log_usage(self.model_name, len(inputs["input_ids"][0]), len(outputs[0]), len(translation.split()))
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

            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            start_time = time.time()
            outputs = self.model.generate(
                **inputs,
                max_length=params["max_tokens"],
                min_length=params["min_tokens"],
                temperature=params["temperature"],
                top_p=params["top_p"],
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self.truncate_output(response, self.max_output_words)
            inference_time = time.time() - start_time
            self.cost_tracker.log_usage(self.model_name, len(inputs["input_ids"][0]), len(outputs[0]), len(response.split()))
            self._save_cache(cache_key, {"response": response, "inference_time": inference_time})
            self.logger.info(f"Generated chat response in {inference_time:.2f}s")
            return response, inference_time
        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            raise

    def get_name(self) -> str:
        return self.model_name