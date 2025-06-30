from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import evaluate
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from utils.logger import setup_logger
import nltk
import warnings
from pathlib import Path
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception:
    pass

class Evaluator:
    """
    Enhanced evaluator with comprehensive metrics for summarization, translation,
    and chat tasks with detailed analysis and reporting capabilities
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
            use_stemmer=True
        )
        
        # Initialize BERTScore
        try:
            self.bertscore = evaluate.load("bertscore")
            self.bertscore_available = True
            self.logger.info("BERTScore initialized successfully")
        except Exception as e:
            self.logger.warning(f"BERTScore not available: {e}")
            self.bertscore_available = False
        
        # Initialize BLEU smoother
        self.bleu_smoother = SmoothingFunction()
        
        # Initialize perplexity models (lightweight alternatives)
        self.perplexity_models = {}
        self._init_perplexity_models()
        
        # Evaluation history
        self.evaluation_history = []
        
        # Metric thresholds for quality assessment
        self.quality_thresholds = {
            "rouge1": {"excellent": 0.6, "good": 0.4, "fair": 0.25, "poor": 0.1},
            "rouge2": {"excellent": 0.4, "good": 0.25, "fair": 0.15, "poor": 0.05},
            "rougeL": {"excellent": 0.5, "good": 0.35, "fair": 0.2, "poor": 0.08},
            "bertscore": {"excellent": 0.85, "good": 0.75, "fair": 0.65, "poor": 0.55},
            "bleu": {"excellent": 0.5, "good": 0.3, "fair": 0.15, "poor": 0.05},
            "meteor": {"excellent": 0.6, "good": 0.4, "fair": 0.25, "poor": 0.1},
            "perplexity": {"excellent": 20, "good": 50, "fair": 100, "poor": 200}  # Lower is better
        }
        
        self.logger.info("Enhanced evaluator initialized successfully")

    def _init_perplexity_models(self):
        """Initialize lightweight models for perplexity calculation"""
        try:
            # Try to use a lightweight model for perplexity
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "distilgpt2"  # Lightweight model
            
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Add padding token if needed
            if self.perplexity_tokenizer.pad_token is None:
                self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token
            
            self.perplexity_available = True
            self.logger.info("Perplexity model initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Perplexity model not available: {e}")
            self.perplexity_available = False

    def evaluate_summarization(self, results: List[Dict], reference: str, 
                             additional_metrics: List[str] = None) -> pd.DataFrame:
        """
        Comprehensive summarization evaluation
        
        Args:
            results: List of model results with outputs
            reference: Reference summary
            additional_metrics: Additional metrics to compute
            
        Returns:
            DataFrame with evaluation results
        """
        try:
            if not reference or not reference.strip():
                raise ValueError("Reference summary cannot be empty")
            
            evaluation_data = []
            
            for result in results:
                model_name = result.get("model", "Unknown")
                output = result.get("output", "")
                inference_time = result.get("inference_time", 0.0)
                
                if not output or not isinstance(output, str):
                    # Handle failed outputs
                    eval_row = {
                        "model": model_name,
                        "rouge1": 0.0,
                        "rouge2": 0.0,
                        "rougeL": 0.0,
                        "rougeLsum": 0.0,
                        "bertscore_f1": 0.0,
                        "inference_time": inference_time,
                        "output_length": 0,
                        "compression_ratio": 0.0,
                        "quality_category": "failed"
                    }
                    evaluation_data.append(eval_row)
                    continue
                
                # ROUGE scores
                rouge_scores = self.rouge_scorer.score(reference, output)
                
                # BERTScore
                bertscore_f1 = 0.0
                if self.bertscore_available:
                    try:
                        bert_results = self.bertscore.compute(
                            predictions=[output],
                            references=[reference],
                            lang="en"
                        )
                        bertscore_f1 = bert_results["f1"][0]
                    except Exception as e:
                        self.logger.debug(f"BERTScore calculation failed: {e}")
                
                # Additional metrics
                output_length = len(output.split())
                reference_length = len(reference.split())
                compression_ratio = output_length / reference_length if reference_length > 0 else 0
                
                # Semantic coherence (basic)
                coherence_score = self._calculate_coherence(output)
                
                # Coverage score (how much of reference is covered)
                coverage_score = self._calculate_coverage(output, reference)
                
                # Redundancy score
                redundancy_score = self._calculate_redundancy(output)
                
                # Overall quality assessment
                quality_category = self._assess_quality("summarization", {
                    "rouge1": rouge_scores["rouge1"].fmeasure,
                    "rouge2": rouge_scores["rouge2"].fmeasure,
                    "rougeL": rouge_scores["rougeL"].fmeasure,
                    "bertscore": bertscore_f1
                })
                
                eval_row = {
                    "model": model_name,
                    "rouge1": rouge_scores["rouge1"].fmeasure,
                    "rouge2": rouge_scores["rouge2"].fmeasure,
                    "rougeL": rouge_scores["rougeL"].fmeasure,
                    "rougeLsum": rouge_scores["rougeLsum"].fmeasure,
                    "bertscore_f1": bertscore_f1,
                    "inference_time": inference_time,
                    "output_length": output_length,
                    "compression_ratio": compression_ratio,
                    "coherence_score": coherence_score,
                    "coverage_score": coverage_score,
                    "redundancy_score": redundancy_score,
                    "quality_category": quality_category
                }
                
                # Add additional metrics if requested
                if additional_metrics:
                    additional_scores = self._calculate_additional_metrics(
                        output, reference, additional_metrics, "summarization"
                    )
                    eval_row.update(additional_scores)
                
                evaluation_data.append(eval_row)
            
            df = pd.DataFrame(evaluation_data)
            
            # Store evaluation in history
            self._store_evaluation("summarization", df.to_dict('records'))
            
            self.logger.info(f"Summarization evaluation completed for {len(results)} models")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in summarization evaluation: {e}")
            raise

    def evaluate_translation(self, results: List[Dict], reference: str,
                           source_language: str = "en", target_language: str = "auto") -> pd.DataFrame:
        """
        Comprehensive translation evaluation
        
        Args:
            results: List of model results with outputs
            reference: Reference translation
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            DataFrame with evaluation results
        """
        try:
            if not reference or not reference.strip():
                raise ValueError("Reference translation cannot be empty")
            
            evaluation_data = []
            
            for result in results:
                model_name = result.get("model", "Unknown")
                output = result.get("output", "")
                inference_time = result.get("inference_time", 0.0)
                
                if not output or not isinstance(output, str):
                    # Handle failed outputs
                    eval_row = {
                        "model": model_name,
                        "bleu": 0.0,
                        "bleu_smooth": 0.0,
                        "meteor": 0.0,
                        "bertscore_f1": 0.0,
                        "inference_time": inference_time,
                        "length_ratio": 0.0,
                        "quality_category": "failed"
                    }
                    evaluation_data.append(eval_row)
                    continue
                
                # BLEU score (standard and smoothed)
                reference_tokens = reference.split()
                output_tokens = output.split()
                
                try:
                    bleu_score = sentence_bleu([reference_tokens], output_tokens)
                    bleu_smooth = sentence_bleu(
                        [reference_tokens], output_tokens,
                        smoothing_function=self.bleu_smoother.method1
                    )
                except Exception as e:
                    self.logger.debug(f"BLEU calculation failed: {e}")
                    bleu_score = 0.0
                    bleu_smooth = 0.0
                
                # METEOR score
                try:
                    meteor_score_val = meteor_score([reference_tokens], output_tokens)
                except Exception as e:
                    self.logger.debug(f"METEOR calculation failed: {e}")
                    meteor_score_val = 0.0
                
                # BERTScore
                bertscore_f1 = 0.0
                if self.bertscore_available:
                    try:
                        bert_results = self.bertscore.compute(
                            predictions=[output],
                            references=[reference],
                            lang="en"  # Adjust based on target language
                        )
                        bertscore_f1 = bert_results["f1"][0]
                    except Exception as e:
                        self.logger.debug(f"BERTScore calculation failed: {e}")
                
                # Translation-specific metrics
                length_ratio = len(output_tokens) / len(reference_tokens) if reference_tokens else 0
                
                # Fluency score (basic n-gram analysis)
                fluency_score = self._calculate_fluency(output)
                
                # Adequacy score (coverage of reference meaning)
                adequacy_score = self._calculate_adequacy(output, reference)
                
                # Translation errors detection
                error_analysis = self._analyze_translation_errors(output, reference)
                
                # Overall quality assessment
                quality_category = self._assess_quality("translation", {
                    "bleu": bleu_score,
                    "meteor": meteor_score_val,
                    "bertscore": bertscore_f1
                })
                
                eval_row = {
                    "model": model_name,
                    "bleu": bleu_score,
                    "bleu_smooth": bleu_smooth,
                    "meteor": meteor_score_val,
                    "bertscore_f1": bertscore_f1,
                    "inference_time": inference_time,
                    "length_ratio": length_ratio,
                    "fluency_score": fluency_score,
                    "adequacy_score": adequacy_score,
                    "quality_category": quality_category,
                    "error_count": error_analysis["error_count"],
                    "error_types": ",".join(error_analysis["error_types"])
                }
                
                evaluation_data.append(eval_row)
            
            df = pd.DataFrame(evaluation_data)
            
            # Store evaluation in history
            self._store_evaluation("translation", df.to_dict('records'))
            
            self.logger.info(f"Translation evaluation completed for {len(results)} models")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in translation evaluation: {e}")
            raise

    def evaluate_chat(self, results: List[Dict], context: List[Dict] = None,
                     human_preferences: Dict = None) -> pd.DataFrame:
        """
        Comprehensive chat evaluation
        
        Args:
            results: List of model results with outputs
            context: Conversation context
            human_preferences: Human preference scores if available
            
        Returns:
            DataFrame with evaluation results
        """
        try:
            evaluation_data = []
            
            for result in results:
                model_name = result.get("model", "Unknown")
                output = result.get("output", "")
                inference_time = result.get("inference_time", 0.0)
                
                if not output or not isinstance(output, str):
                    # Handle failed outputs
                    eval_row = {
                        "model": model_name,
                        "perplexity": float("inf"),
                        "response_length": 0,
                        "inference_time": inference_time,
                        "quality_category": "failed"
                    }
                    evaluation_data.append(eval_row)
                    continue
                
                # Perplexity calculation
                perplexity = self._calculate_perplexity(output)
                
                # Response quality metrics
                response_length = len(output.split())
                
                # Relevance to context
                relevance_score = self._calculate_relevance(output, context) if context else 0.5
                
                # Coherence and fluency
                coherence_score = self._calculate_coherence(output)
                fluency_score = self._calculate_fluency(output)
                
                # Engagement score
                engagement_score = self._calculate_engagement(output)
                
                # Safety score
                safety_score = self._calculate_safety(output)
                
                # Informativeness
                informativeness_score = self._calculate_informativeness(output)
                
                # Human preference alignment
                preference_score = 0.5
                if human_preferences and model_name in human_preferences:
                    preference_score = human_preferences[model_name]
                
                # Overall quality assessment
                quality_category = self._assess_quality("chat", {
                    "perplexity": perplexity,
                    "relevance": relevance_score,
                    "coherence": coherence_score,
                    "safety": safety_score
                })
                
                eval_row = {
                    "model": model_name,
                    "perplexity": perplexity,
                    "response_length": response_length,
                    "relevance_score": relevance_score,
                    "coherence_score": coherence_score,
                    "fluency_score": fluency_score,
                    "engagement_score": engagement_score,
                    "safety_score": safety_score,
                    "informativeness_score": informativeness_score,
                    "preference_score": preference_score,
                    "inference_time": inference_time,
                    "quality_category": quality_category
                }
                
                evaluation_data.append(eval_row)
            
            df = pd.DataFrame(evaluation_data)
            
            # Store evaluation in history
            self._store_evaluation("chat", df.to_dict('records'))
            
            self.logger.info(f"Chat evaluation completed for {len(results)} models")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in chat evaluation: {e}")
            raise

    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity using lightweight model"""
        try:
            if not self.perplexity_available or not text.strip():
                return float("inf")
            
            # Tokenize text
            inputs = self.perplexity_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            # Calculate loss
            with torch.no_grad():
                outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            # Convert loss to perplexity
            perplexity = torch.exp(loss).item()
            
            return min(perplexity, 1000.0)  # Cap at 1000 for sanity
            
        except Exception as e:
            self.logger.debug(f"Perplexity calculation failed: {e}")
            return float("inf")

    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score"""
        try:
            if not text.strip():
                return 0.0
            
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 2:
                return 0.7  # Single sentence gets moderate score
            
            # Basic coherence based on sentence transitions
            coherence_indicators = [
                'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                'consequently', 'meanwhile', 'subsequently', 'nevertheless',
                'thus', 'hence', 'accordingly', 'likewise', 'similarly'
            ]
            
            text_lower = text.lower()
            indicator_count = sum(1 for indicator in coherence_indicators if indicator in text_lower)
            
            # Score based on presence of coherence indicators and sentence structure
            base_score = 0.5
            indicator_score = min(0.3, indicator_count * 0.1)
            length_score = min(0.2, len(sentences) * 0.05)
            
            return min(1.0, base_score + indicator_score + length_score)
            
        except Exception as e:
            self.logger.debug(f"Coherence calculation failed: {e}")
            return 0.5

    def _calculate_coverage(self, output: str, reference: str) -> float:
        """Calculate how well output covers reference content"""
        try:
            if not output or not reference:
                return 0.0
            
            output_words = set(output.lower().split())
            reference_words = set(reference.lower().split())
            
            if not reference_words:
                return 0.0
            
            covered_words = output_words.intersection(reference_words)
            coverage = len(covered_words) / len(reference_words)
            
            return min(1.0, coverage)
            
        except Exception as e:
            self.logger.debug(f"Coverage calculation failed: {e}")
            return 0.0

    def _calculate_redundancy(self, text: str) -> float:
        """Calculate redundancy in text (lower is better)"""
        try:
            if not text.strip():
                return 1.0
            
            words = text.lower().split()
            if len(words) < 5:
                return 0.0
            
            unique_words = set(words)
            redundancy = 1.0 - (len(unique_words) / len(words))
            
            return redundancy
            
        except Exception as e:
            self.logger.debug(f"Redundancy calculation failed: {e}")
            return 0.5

    def _calculate_fluency(self, text: str) -> float:
        """Calculate text fluency score"""
        try:
            if not text.strip():
                return 0.0
            
            # Basic fluency indicators
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            words = text.split()
            
            if not sentences or not words:
                return 0.0
            
            # Average sentence length (moderate is good)
            avg_sentence_length = len(words) / len(sentences)
            length_score = 1.0 - abs(avg_sentence_length - 15) / 15  # Optimal around 15 words
            length_score = max(0.0, min(1.0, length_score))
            
            # Grammar indicators (basic)
            grammar_score = 0.7  # Default moderate score
            
            # Check for obvious errors
            if text.count(',,') > 0 or text.count('..') > text.count('...'):
                grammar_score -= 0.2
            
            # Capitalization check
            if text[0].islower() if text else False:
                grammar_score -= 0.1
            
            return (length_score + grammar_score) / 2
            
        except Exception as e:
            self.logger.debug(f"Fluency calculation failed: {e}")
            return 0.5

    def _calculate_adequacy(self, output: str, reference: str) -> float:
        """Calculate translation adequacy (meaning preservation)"""
        try:
            # Use coverage as a proxy for adequacy
            return self._calculate_coverage(output, reference)
            
        except Exception as e:
            self.logger.debug(f"Adequacy calculation failed: {e}")
            return 0.5

    def _analyze_translation_errors(self, output: str, reference: str) -> Dict:
        """Analyze common translation errors"""
        try:
            errors = []
            error_count = 0
            
            # Basic error detection
            if len(output) < len(reference) * 0.5:
                errors.append("undertranslation")
                error_count += 1
            
            if len(output) > len(reference) * 2:
                errors.append("overtranslation")
                error_count += 1
            
            # Check for obvious errors
            if '...' in output and '...' not in reference:
                errors.append("incomplete")
                error_count += 1
            
            return {
                "error_count": error_count,
                "error_types": errors
            }
            
        except Exception as e:
            self.logger.debug(f"Error analysis failed: {e}")
            return {"error_count": 0, "error_types": []}

    def _calculate_relevance(self, output: str, context: List[Dict]) -> float:
        """Calculate relevance to conversation context"""
        try:
            if not context or not output:
                return 0.5
            
            # Extract context words
            context_text = " ".join([msg.get("content", "") for msg in context])
            context_words = set(context_text.lower().split())
            output_words = set(output.lower().split())
            
            if not context_words:
                return 0.5
            
            # Calculate overlap
            overlap = len(context_words.intersection(output_words))
            relevance = overlap / max(len(context_words), len(output_words))
            
            return min(1.0, relevance * 2)  # Scale up since perfect overlap is rare
            
        except Exception as e:
            self.logger.debug(f"Relevance calculation failed: {e}")
            return 0.5

    def _calculate_engagement(self, text: str) -> float:
        """Calculate engagement score"""
        try:
            if not text.strip():
                return 0.0
            
            engagement_indicators = [
                '?', '!', 'you', 'your', 'what', 'how', 'why', 'when', 'where',
                'think', 'feel', 'believe', 'opinion', 'interesting', 'amazing'
            ]
            
            text_lower = text.lower()
            indicator_count = sum(1 for indicator in engagement_indicators if indicator in text_lower)
            
            # Normalize by text length
            words = text.split()
            engagement_ratio = indicator_count / max(len(words), 1)
            
            return min(1.0, engagement_ratio * 10)  # Scale appropriately
            
        except Exception as e:
            self.logger.debug(f"Engagement calculation failed: {e}")
            return 0.5

    def _calculate_safety(self, text: str) -> float:
        """Calculate safety score (1.0 = very safe)"""
        try:
            if not text.strip():
                return 1.0
            
            # Basic safety indicators (inverse of harmful content)
            harmful_patterns = [
                'hate', 'kill', 'violence', 'threat', 'harm', 'dangerous',
                'illegal', 'drug', 'weapon', 'bomb', 'terrorist'
            ]
            
            text_lower = text.lower()
            harmful_count = sum(1 for pattern in harmful_patterns if pattern in text_lower)
            
            # Safety decreases with harmful content
            safety_score = max(0.0, 1.0 - (harmful_count * 0.2))
            
            return safety_score
            
        except Exception as e:
            self.logger.debug(f"Safety calculation failed: {e}")
            return 0.8  # Default to mostly safe

    def _calculate_informativeness(self, text: str) -> float:
        """Calculate informativeness score"""
        try:
            if not text.strip():
                return 0.0
            
            # Indicators of informative content
            info_indicators = [
                'because', 'therefore', 'research', 'study', 'evidence', 'data',
                'fact', 'information', 'according', 'analysis', 'result',
                'percent', '%', 'number', 'statistics', 'report'
            ]
            
            text_lower = text.lower()
            info_count = sum(1 for indicator in info_indicators if indicator in text_lower)
            
            # Vocabulary diversity
            words = text.split()
            unique_words = set(text_lower.split())
            diversity = len(unique_words) / max(len(words), 1)
            
            # Combined score
            info_score = min(1.0, (info_count * 0.1) + diversity)
            
            return info_score
            
        except Exception as e:
            self.logger.debug(f"Informativeness calculation failed: {e}")
            return 0.5

    def _assess_quality(self, task_type: str, scores: Dict[str, float]) -> str:
        """Assess overall quality category"""
        try:
            if not scores:
                return "unknown"
            
            # Get relevant thresholds for task
            if task_type == "summarization":
                key_metrics = ["rouge1", "rouge2", "rougeL", "bertscore"]
            elif task_type == "translation":
                key_metrics = ["bleu", "meteor", "bertscore"]
            elif task_type == "chat":
                key_metrics = ["perplexity", "relevance", "coherence", "safety"]
            else:
                key_metrics = list(scores.keys())
            
            # Calculate average quality
            quality_scores = []
            for metric in key_metrics:
                if metric in scores and metric in self.quality_thresholds:
                    score = scores[metric]
                    thresholds = self.quality_thresholds[metric]
                    
                    if metric == "perplexity":
                        # Lower is better for perplexity
                        if score <= thresholds["excellent"]:
                            quality_scores.append(4)
                        elif score <= thresholds["good"]:
                            quality_scores.append(3)
                        elif score <= thresholds["fair"]:
                            quality_scores.append(2)
                        else:
                            quality_scores.append(1)
                    else:
                        # Higher is better for other metrics
                        if score >= thresholds["excellent"]:
                            quality_scores.append(4)
                        elif score >= thresholds["good"]:
                            quality_scores.append(3)
                        elif score >= thresholds["fair"]:
                            quality_scores.append(2)
                        else:
                            quality_scores.append(1)
            
            if not quality_scores:
                return "unknown"
            
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            if avg_quality >= 3.5:
                return "excellent"
            elif avg_quality >= 2.5:
                return "good"
            elif avg_quality >= 1.5:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            self.logger.debug(f"Quality assessment failed: {e}")
            return "unknown"

    def _calculate_additional_metrics(self, output: str, reference: str, 
                                    metrics: List[str], task_type: str) -> Dict[str, float]:
        """Calculate additional metrics based on request"""
        additional_scores = {}
        
        for metric in metrics:
            try:
                if metric == "sentence_count":
                    additional_scores[metric] = len([s for s in output.split('.') if s.strip()])
                elif metric == "word_count":
                    additional_scores[metric] = len(output.split())
                elif metric == "char_count":
                    additional_scores[metric] = len(output)
                elif metric == "unique_words":
                    additional_scores[metric] = len(set(output.lower().split()))
                elif metric == "avg_word_length":
                    words = output.split()
                    additional_scores[metric] = sum(len(w) for w in words) / max(len(words), 1)
                # Add more custom metrics as needed
            except Exception as e:
                self.logger.debug(f"Additional metric {metric} calculation failed: {e}")
                additional_scores[metric] = 0.0
        
        return additional_scores

    def _store_evaluation(self, task_type: str, evaluation_data: List[Dict]):
        """Store evaluation in history"""
        try:
            evaluation_record = {
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
                "model_count": len(evaluation_data),
                "data": evaluation_data
            }
            
            self.evaluation_history.append(evaluation_record)
            
            # Keep only recent evaluations
            if len(self.evaluation_history) > 100:
                self.evaluation_history = self.evaluation_history[-100:]
                
        except Exception as e:
            self.logger.debug(f"Error storing evaluation: {e}")

    def compare_evaluations(self, evaluation1: pd.DataFrame, evaluation2: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare two evaluation results
        
        Args:
            evaluation1: First evaluation DataFrame
            evaluation2: Second evaluation DataFrame
            
        Returns:
            Comparison results
        """
        try:
            if evaluation1.empty or evaluation2.empty:
                return {"error": "One or both evaluations are empty"}
            
            # Find common metrics
            metrics1 = set(evaluation1.columns) - {"model", "inference_time"}
            metrics2 = set(evaluation2.columns) - {"model", "inference_time"}
            common_metrics = metrics1.intersection(metrics2)
            
            if not common_metrics:
                return {"error": "No common metrics found between evaluations"}
            
            comparison = {
                "common_metrics": list(common_metrics),
                "evaluation1_models": evaluation1["model"].tolist(),
                "evaluation2_models": evaluation2["model"].tolist(),
                "metric_comparison": {}
            }
            
            # Compare each metric
            for metric in common_metrics:
                metric_comp = {
                    "eval1_mean": evaluation1[metric].mean(),
                    "eval2_mean": evaluation2[metric].mean(),
                    "eval1_std": evaluation1[metric].std(),
                    "eval2_std": evaluation2[metric].std(),
                    "difference": evaluation2[metric].mean() - evaluation1[metric].mean()
                }
                comparison["metric_comparison"][metric] = metric_comp
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing evaluations: {e}")
            return {"error": str(e)}