from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from utils.logger import setup_logger
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class Evaluator:
    """
    Working evaluator that returns proper DataFrames for visualization
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # BLEU smoother
        self.bleu_smoother = SmoothingFunction()
        
        self.logger.info("Evaluator initialized")

    def evaluate_summarization(self, results: List[Dict], reference: str) -> Optional[pd.DataFrame]:
        """
        Evaluate summarization results and return DataFrame for graphs
        
        Args:
            results: List of model results
            reference: Reference summary
            
        Returns:
            DataFrame with evaluation metrics
        """
        try:
            evaluation_data = []
            
            for result in results:
                model_name = result.get("model", "Unknown")
                output = result.get("output", "")
                inference_time = result.get("inference_time", 0.0)
                
                if not output:
                    # Add default values for failed models
                    eval_row = {
                        "model": model_name,
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                        "inference_time": inference_time,
                        "word_count": 0,
                        "success": False
                    }
                else:
                    # Calculate ROUGE scores
                    rouge_scores = self.rouge_scorer.score(reference, output)
                    
                    eval_row = {
                        "model": model_name,
                        "rouge1": rouge_scores["rouge1"].fmeasure,
                        "rouge2": rouge_scores["rouge2"].fmeasure,
                        "rougeL": rouge_scores["rougeL"].fmeasure,
                        "inference_time": inference_time,
                        "word_count": len(output.split()),
                        "success": True
                    }
                
                evaluation_data.append(eval_row)
            
            df = pd.DataFrame(evaluation_data)
            self.logger.info(f"Evaluation completed for {len(results)} models")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in summarization evaluation: {e}")
            return None

    def evaluate_translation(self, results: List[Dict], reference: str) -> Optional[pd.DataFrame]:
        """
        Evaluate translation results and return DataFrame for graphs
        
        Args:
            results: List of model results
            reference: Reference translation
            
        Returns:
            DataFrame with evaluation metrics
        """
        try:
            evaluation_data = []
            
            for result in results:
                model_name = result.get("model", "Unknown")
                output = result.get("output", "")
                inference_time = result.get("inference_time", 0.0)
                
                if not output:
                    eval_row = {
                        "model": model_name,
                        "bleu": 0.0,
                        "rouge1": 0.0,
                        "inference_time": inference_time,
                        "word_count": 0,
                        "success": False
                    }
                else:
                    # Calculate BLEU score
                    reference_tokens = reference.split()
                    output_tokens = output.split()
                    
                    try:
                        bleu_score = sentence_bleu([reference_tokens], output_tokens)
                    except:
                        bleu_score = 0.0
                    
                    # Calculate ROUGE for additional comparison
                    rouge_scores = self.rouge_scorer.score(reference, output)
                    
                    eval_row = {
                        "model": model_name,
                        "bleu": bleu_score,
                        "rouge1": rouge_scores["rouge1"].fmeasure,
                        "inference_time": inference_time,
                        "word_count": len(output.split()),
                        "success": True
                    }
                
                evaluation_data.append(eval_row)
            
            df = pd.DataFrame(evaluation_data)
            self.logger.info(f"Translation evaluation completed for {len(results)} models")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in translation evaluation: {e}")
            return None
