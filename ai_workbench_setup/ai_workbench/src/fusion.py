from utils.logger import setup_logger
from typing import List, Dict, Any
import pandas as pd
from rouge_score import rouge_scorer

class ModelFusion:
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def fuse_outputs(self, results: List[Dict], reference: str = None) -> Dict:
        try:
            if not results or all(r["output"] is None for r in results):
                return {"fused_output": None, "weights": {}}

            outputs = [r["output"] for r in results if r["output"]]
            models = [r["model"] for r in results if r["output"]]
            if len(outputs) < 2:
                return {"fused_output": outputs[0] if outputs else None, "weights": {models[0]: 1.0} if models else {}}

            weights = {}
            if reference:
                for model, output in zip(models, outputs):
                    score = self.rouge_scorer.score(reference, output)["rougeL"].fmeasure
                    weights[model] = score
                total = sum(weights.values())
                if total > 0:
                    weights = {k: v / total for k, v in weights.items()}
                else:
                    weights = {model: 1.0 / len(models) for model in models}
            else:
                weights = {model: 1.0 / len(models) for model in models}

            fused_output = " ".join(outputs)[:100]
            self.logger.info("Fused outputs with weights: %s", weights)
            return {"fused_output": fused_output, "weights": weights}
        except Exception as e:
            self.logger.error(f"Error in model fusion: {str(e)}")
            raise