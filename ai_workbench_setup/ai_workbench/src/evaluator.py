from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from utils.logger import setup_logger
from typing import List, Dict
import nltk
nltk.download('wordnet')

class Evaluator:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bertscore = evaluate.load("bertscore")
        self.logger.info("Evaluator initialized")

    def evaluate_summarization(self, results: List[Dict], reference: str) -> pd.DataFrame:
        try:
            evaluation = []
            for result in results:
                if result["output"] is None:
                    evaluation.append({
                        "model": result["model"],
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                        "bertscore": 0.0,
                        "inference_time": result["inference_time"] or 0.0
                    })
                    continue
                rouge_scores = self.rouge_scorer.score(reference, result["output"])
                bertscore = self.bertscore.compute(
                    predictions=[result["output"]],
                    references=[reference],
                    lang="en"
                )["f1"][0]
                evaluation.append({
                    "model": result["model"],
                    "rouge1": rouge_scores["rouge1"].fmeasure,
                    "rouge2": rouge_scores["rouge2"].fmeasure,
                    "rougeL": rouge_scores["rougeL"].fmeasure,
                    "bertscore": bertscore,
                    "inference_time": result["inference_time"]
                })
            df = pd.DataFrame(evaluation)
            self.logger.info("Summarization evaluation completed")
            return df
        except Exception as e:
            self.logger.error(f"Error during summarization evaluation: {str(e)}")
            raise

    def evaluate_translation(self, results: List[Dict], reference: str) -> pd.DataFrame:
        try:
            evaluation = []
            for result in results:
                if result["output"] is None:
                    evaluation.append({
                        "model": result["model"],
                        "bleu": 0.0,
                        "meteor": 0.0,
                        "inference_time": result["inference_time"] or 0.0
                    })
                    continue
                bleu = sentence_bleu([reference.split()], result["output"].split())
                meteor = meteor_score([reference.split()], result["output"].split())
                evaluation.append({
                    "model": result["model"],
                    "bleu": bleu,
                    "meteor": meteor,
                    "inference_time": result["inference_time"]
                })
            df = pd.DataFrame(evaluation)
            self.logger.info("Translation evaluation completed")
            return df
        except Exception as e:
            self.logger.error(f"Error during translation evaluation: {str(e)}")
            raise

    def evaluate_chat(self, results: List[Dict], model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> pd.DataFrame:
        try:
            evaluation = []
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name).eval()
            for result in results:
                if result["output"] is None:
                    evaluation.append({
                        "model": result["model"],
                        "perplexity": float("inf"),
                        "inference_time": result["inference_time"] or 0.0
                    })
                    continue
                inputs = tokenizer(result["output"], return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    perplexity = torch.exp(outputs.loss).item()
                evaluation.append({
                    "model": result["model"],
                    "perplexity": perplexity,
                    "inference_time": result["inference_time"]
                })
            df = pd.DataFrame(evaluation)
            self.logger.info("Chat evaluation completed")
            return df
        except Exception as e:
            self.logger.error(f"Error during chat evaluation: {str(e)}")
            raise

    def suggest_metrics(self, task: str) -> Dict:
        suggestions = {
            "summarization": {
                "metrics": ["rouge1", "rouge2", "rougeL", "bertscore"],
                "reason": "ROUGE measures n-gram overlap, good for structure; BERTScore captures semantic similarity."
            },
            "translation": {
                "metrics": ["bleu", "meteor"],
                "reason": "BLEU is standard for n-gram precision; METEOR considers synonyms and stemming."
            },
            "chat": {
                "metrics": ["perplexity"],
                "reason": "Perplexity measures response coherence; user ratings can add subjective quality."
            }
        }
        return suggestions.get(task, {"metrics": [], "reason": "No suggestions for this task."})