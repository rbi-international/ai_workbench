from utils.logger import setup_logger
import pandas as pd
from typing import List, Dict, Any

class AITutor:
    def __init__(self):
        self.logger = setup_logger(__name__)

    def explain_performance(self, task: str, results: List[Dict], evaluation: pd.DataFrame) -> str:
        try:
            if evaluation is None or evaluation.empty:
                return "No evaluation data available to compare performance."

            explanations = []
            models = evaluation["model"].tolist()
            metrics = [col for col in evaluation.columns if col != "model"]

            for metric in metrics:
                best_model = evaluation.loc[evaluation[metric].idxmax(), "model"]
                best_score = evaluation[metric].max()
                reasons = []
                if task == "summarization":
                    if metric in ["rouge1", "rouge2", "rougeL"]:
                        reasons.append(f"{best_model} achieved the highest {metric.upper()} ({best_score:.3f}), indicating better n-gram overlap with the reference, likely due to precise phrasing.")
                    elif metric == "bertscore":
                        reasons.append(f"{best_model}'s BERTScore ({best_score:.3f}) suggests stronger semantic similarity, possibly from better context understanding.")
                elif task == "translation":
                    if metric == "bleu":
                        reasons.append(f"{best_model}'s BLEU score ({best_score:.3f}) reflects higher precision in word choice, capturing the reference's structure.")
                    elif metric == "meteor":
                        reasons.append(f"{best_model}'s METEOR score ({best_score:.3f}) indicates better fluency, considering synonyms and stemming.")
                elif task == "chat":
                    if metric == "perplexity":
                        reasons.append(f"{best_model}'s lower perplexity ({best_score:.3f}) suggests more coherent and contextually relevant responses.")
                if "inference_time" in evaluation.columns:
                    fastest_model = evaluation.loc[evaluation["inference_time"].idxmin(), "model"]
                    fastest_time = evaluation["inference_time"].min()
                    reasons.append(f"{fastest_model} was fastest (time: {fastest_time:.2f}s), likely due to optimized architecture or API efficiency.")
                explanations.append(f"For {metric.upper()}: {', '.join(reasons)}")

            for i, model in enumerate(models):
                output = next((r["output"] for r in results if r["model"] == model), None)
                if output:
                    length = len(output.split())
                    explanations.append(f"{model}'s output (length: {length} words) was {'concise' if length < 50 else 'detailed'}. Sample: {output[:50]}...")

            self.logger.info("Generated performance explanation for task: %s", task)
            return "\n".join(explanations)
        except Exception as e:
            self.logger.error(f"Error in AI Tutor explanation: {str(e)}")
            return f"Error generating explanation: {str(e)}"