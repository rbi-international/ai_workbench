from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from detoxify import Detoxify
from utils.logger import setup_logger
import yaml
from typing import Dict, List
import os
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

class EthicsAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Explicitly load unitary/toxic-bert with token
        try:
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token:
                self.logger.warning("HUGGINGFACE_TOKEN not found in .env, attempting without token")
            self.toxicity_analyzer = Detoxify(model_type='original', device='cpu')
        except Exception as e:
            self.logger.error(f"Failed to initialize Detoxify: {str(e)}")
            # Fallback to manual model loading
            self.logger.info("Falling back to manual unitary/toxic-bert loading")
            model_name = "unitary/toxic-bert"
            self.toxicity_analyzer = {
                'model': AutoModelForSequenceClassification.from_pretrained(model_name, token=token),
                'tokenizer': AutoTokenizer.from_pretrained(model_name, token=token),
                'predict': lambda text: self._manual_toxicity_predict(text)
            }
        self.sentiment_threshold = self.config["ethics"]["sentiment_threshold"]
        self.toxicity_threshold = self.config["ethics"]["toxicity_threshold"]

    def _manual_toxicity_predict(self, text: str) -> Dict:
        try:
            inputs = self.toxicity_analyzer['tokenizer'](text, return_tensors="pt", truncation=True, padding=True)
            outputs = self.toxicity_analyzer['model'](**inputs)
            scores = outputs.logits.softmax(dim=-1).detach().numpy()[0]
            labels = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_hate']
            return {label: float(score) for label, score in zip(labels, scores)}
        except Exception as e:
            self.logger.error(f"Error in manual toxicity prediction: {str(e)}")
            return {}

    def analyze(self, outputs: List[str]) -> List[Dict]:
        try:
            results = []
            for output in outputs:
                if not output:
                    results.append({"sentiment": {}, "toxicity": {}, "warnings": []})
                    continue

                sentiment = self.sentiment_analyzer.polarity_scores(output)
                sentiment_warning = []
                if abs(sentiment["compound"]) > self.sentiment_threshold:
                    sentiment_warning.append(f"Strong sentiment detected (compound: {sentiment['compound']:.2f})")

                toxicity = self.toxicity_analyzer.predict(output) if hasattr(self.toxicity_analyzer, 'predict') else self._manual_toxicity_predict(output)
                toxicity_warning = []
                for key, value in toxicity.items():
                    if value > self.toxicity_threshold:
                        toxicity_warning.append(f"High {key} score: {value:.2f}")

                results.append({
                    "sentiment": sentiment,
                    "toxicity": toxicity,
                    "warnings": sentiment_warning + toxicity_warning
                })
            self.logger.info("Completed ethics analysis")
            return results
        except Exception as e:
            self.logger.error(f"Error in ethics analysis: {str(e)}")
            raise