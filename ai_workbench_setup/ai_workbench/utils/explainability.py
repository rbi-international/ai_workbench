import shap
from transformers import pipeline
from utils.logger import setup_logger

class Explainability:
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.explainer = shap.Explainer(pipeline("text-generation", model="distilgpt2"))

    def explain(self, output: str, input_text: str) -> dict:
        try:
            shap_values = self.explainer([input_text])
            return {
                "input": input_text,
                "output": output,
                "shap_values": shap_values.values.tolist()
            }
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            return {}