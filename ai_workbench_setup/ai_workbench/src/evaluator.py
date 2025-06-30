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
import re
from collections import Counter
import textstat

# Suppress warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass


class ComprehensiveEvaluator:
    """
    Enhanced evaluator with comprehensive metrics for all model types and tasks
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        
        # Initialize all scorers and metrics
        self._init_scorers()
        
        # Load comprehensive metric definitions
        self.metric_definitions = self._load_metric_definitions()
        
        # Evaluation history
        self.evaluation_history = []
        
        self.logger.info("Comprehensive evaluator initialized with all metrics")

    def _init_scorers(self):
        """Initialize all scoring systems"""
        try:
            # ROUGE scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], 
                use_stemmer=True
            )
            
            # BERTScore
            try:
                self.bertscore = evaluate.load("bertscore")
                self.bertscore_available = True
                self.logger.info("BERTScore initialized")
            except Exception as e:
                self.logger.warning(f"BERTScore not available: {e}")
                self.bertscore_available = False
            
            # BLEU smoother
            self.bleu_smoother = SmoothingFunction()
            
            # Sentiment analyzer
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                self.sentiment_available = True
            except ImportError:
                self.sentiment_available = False
                self.logger.warning("VADER sentiment analyzer not available")
            
            # Initialize perplexity models
            self.perplexity_models = {}
            self._init_perplexity_models()
            
        except Exception as e:
            self.logger.error(f"Error initializing scorers: {e}")

    def _init_perplexity_models(self):
        """Initialize lightweight models for perplexity calculation"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "distilgpt2"  # Lightweight model
            
            self.perplexity_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.perplexity_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.perplexity_tokenizer.pad_token is None:
                self.perplexity_tokenizer.pad_token = self.perplexity_tokenizer.eos_token
            
            self.perplexity_available = True
            self.logger.info("Perplexity model initialized")
            
        except Exception as e:
            self.logger.warning(f"Perplexity model not available: {e}")
            self.perplexity_available = False

    def _load_metric_definitions(self) -> Dict[str, Dict]:
        """Load comprehensive metric definitions"""
        return {
            # Text Quality Metrics
            "rouge1": {
                "name": "ROUGE-1",
                "description": "Unigram overlap between generated and reference text",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.4,
                "category": "Text Quality",
                "task_types": ["summarization", "generation"]
            },
            "rouge2": {
                "name": "ROUGE-2", 
                "description": "Bigram overlap between generated and reference text",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.25,
                "category": "Text Quality",
                "task_types": ["summarization", "generation"]
            },
            "rougeL": {
                "name": "ROUGE-L",
                "description": "Longest common subsequence between texts",
                "range": "0-1 (higher is better)", 
                "good_threshold": 0.35,
                "category": "Text Quality",
                "task_types": ["summarization", "generation"]
            },
            "bertscore_f1": {
                "name": "BERTScore F1",
                "description": "Semantic similarity using BERT embeddings",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.7,
                "category": "Semantic Quality",
                "task_types": ["summarization", "translation", "generation"]
            },
            "bertscore_precision": {
                "name": "BERTScore Precision",
                "description": "Precision of semantic similarity",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.7,
                "category": "Semantic Quality",
                "task_types": ["summarization", "translation", "generation"]
            },
            "bertscore_recall": {
                "name": "BERTScore Recall",
                "description": "Recall of semantic similarity",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.7,
                "category": "Semantic Quality",
                "task_types": ["summarization", "translation", "generation"]
            },
            "bleu": {
                "name": "BLEU",
                "description": "N-gram precision for translation quality",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.3,
                "category": "Translation Quality",
                "task_types": ["translation"]
            },
            "meteor": {
                "name": "METEOR",
                "description": "Translation evaluation with word order and synonyms",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.4,
                "category": "Translation Quality",
                "task_types": ["translation"]
            },
            "chrf": {
                "name": "chrF",
                "description": "Character-level F-score for translation",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.5,
                "category": "Translation Quality",
                "task_types": ["translation"]
            },
            
            # Performance Metrics
            "inference_time": {
                "name": "Inference Time",
                "description": "Time taken to generate response",
                "range": "0+ seconds (lower is better)",
                "good_threshold": 3.0,
                "category": "Performance",
                "task_types": ["all"]
            },
            "tokens_per_second": {
                "name": "Tokens/Second",
                "description": "Generation speed in tokens per second",
                "range": "0+ (higher is better)",
                "good_threshold": 50,
                "category": "Performance",
                "task_types": ["all"]
            },
            "throughput": {
                "name": "Throughput",
                "description": "Requests processed per minute",
                "range": "0+ req/min (higher is better)",
                "good_threshold": 60,
                "category": "Performance",
                "task_types": ["all"]
            },
            
            # Content Analysis Metrics
            "word_count": {
                "name": "Word Count",
                "description": "Number of words in generated text",
                "range": "0+ words",
                "good_threshold": 50,
                "category": "Content Analysis",
                "task_types": ["all"]
            },
            "sentence_count": {
                "name": "Sentence Count", 
                "description": "Number of sentences in generated text",
                "range": "0+ sentences",
                "good_threshold": 3,
                "category": "Content Analysis",
                "task_types": ["all"]
            },
            "vocabulary_diversity": {
                "name": "Vocabulary Diversity",
                "description": "Ratio of unique words to total words",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.7,
                "category": "Content Analysis",
                "task_types": ["all"]
            },
            "avg_sentence_length": {
                "name": "Avg Sentence Length",
                "description": "Average words per sentence",
                "range": "0+ words",
                "good_threshold": 15,
                "category": "Content Analysis",
                "task_types": ["all"]
            },
            "compression_ratio": {
                "name": "Compression Ratio",
                "description": "Ratio of output to input length",
                "range": "0+ (context dependent)",
                "good_threshold": 0.3,
                "category": "Content Analysis",
                "task_types": ["summarization"]
            },
            
            # Advanced Quality Metrics
            "perplexity": {
                "name": "Perplexity",
                "description": "Model uncertainty/confidence (lower is better)",
                "range": "1+ (lower is better)",
                "good_threshold": 50,
                "category": "Advanced Quality",
                "task_types": ["generation", "chat"]
            },
            "coherence_score": {
                "name": "Coherence Score",
                "description": "Logical flow and consistency of text",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.7,
                "category": "Advanced Quality",
                "task_types": ["all"]
            },
            "relevance_score": {
                "name": "Relevance Score",
                "description": "How well response addresses input",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.8,
                "category": "Advanced Quality",
                "task_types": ["all"]
            },
            "fluency_score": {
                "name": "Fluency Score",
                "description": "Grammar and readability quality",
                "range": "0-1 (higher is better)",
                "good_threshold": 0.8,
                "category": "Advanced Quality",
                "task_types": ["all"]
            },
            "readability_score": {
                "name": "Readability Score",
                "description": "Flesch reading ease score",
                "range": "0-100 (higher is easier to read)",
                "good_threshold": 60,
                "category": "Advanced Quality",
                "task_types": ["all"]
            },
            
            # Sentiment and Style Metrics
            "sentiment_score": {
                "name": "Sentiment Score",
                "description": "Overall sentiment of the text",
                "range": "-1 to 1 (negative to positive)",
                "good_threshold": 0.0,
                "category": "Sentiment Analysis",
                "task_types": ["all"]
            },
            "sentiment_magnitude": {
                "name": "Sentiment Magnitude",
                "description": "Strength of sentiment expression",
                "range": "0-1 (higher is stronger)",
                "good_threshold": 0.5,
                "category": "Sentiment Analysis",
                "task_types": ["all"]
            },
            "formality_score": {
                "name": "Formality Score",
                "description": "Level of formal language use",
                "range": "0-1 (higher is more formal)",
                "good_threshold": 0.5,
                "category": "Style Analysis",
                "task_types": ["all"]
            },
            
            # Task-Specific Metrics
            "factuality_score": {
                "name": "Factuality Score",
                "description": "Estimated factual accuracy",
                "range": "0-1 (higher is more factual)",
                "good_threshold": 0.8,
                "category": "Content Quality",
                "task_types": ["summarization", "chat"]
            },
            "coverage_score": {
                "name": "Coverage Score",
                "description": "How much of input content is covered",
                "range": "0-1 (higher is better coverage)",
                "good_threshold": 0.7,
                "category": "Content Quality",
                "task_types": ["summarization"]
            },
            "novelty_score": {
                "name": "Novelty Score",
                "description": "Amount of new information generated",
                "range": "0-1 (higher is more novel)",
                "good_threshold": 0.3,
                "category": "Content Quality",
                "task_types": ["generation", "chat"]
            }
        }

    def evaluate_comprehensive(self, results: List[Dict], reference: str = None, 
                             task_type: str = "general", input_text: str = None,
                             custom_metrics: List[str] = None) -> pd.DataFrame:
        """
        Comprehensive evaluation with all available metrics
        
        Args:
            results: List of model results
            reference: Reference text for comparison
            task_type: Type of task being evaluated
            input_text: Original input text
            custom_metrics: Specific metrics to calculate
            
        Returns:
            DataFrame with comprehensive evaluation results
        """
        try:
            self.logger.info(f"Starting comprehensive evaluation for {len(results)} models")
            
            evaluation_data = []
            
            for result in results:
                model_name = result.get("model", "Unknown")
                output = result.get("output", "")
                inference_time = result.get("inference_time", 0.0)
                
                self.logger.debug(f"Evaluating {model_name}")
                
                # Initialize evaluation row
                eval_row = {
                    "model": model_name,
                    "inference_time": inference_time,
                    "success": bool(output and output.strip())
                }
                
                if not output or not isinstance(output, str):
                    # Handle failed outputs with zeros
                    eval_row.update(self._get_default_metrics())
                    eval_row["quality_category"] = "failed"
                    evaluation_data.append(eval_row)
                    continue
                
                # Calculate all applicable metrics
                metrics = self._calculate_all_metrics(
                    output, reference, task_type, input_text, inference_time
                )
                
                eval_row.update(metrics)
                
                # Determine quality category
                eval_row["quality_category"] = self._assess_overall_quality(metrics, task_type)
                
                evaluation_data.append(eval_row)
            
            # Create DataFrame
            df = pd.DataFrame(evaluation_data)
            
            # Filter metrics if custom list provided
            if custom_metrics:
                available_metrics = ["model", "inference_time", "success", "quality_category"] + custom_metrics
                df = df[[col for col in available_metrics if col in df.columns]]
            
            # Store evaluation in history
            self._store_evaluation(task_type, df.to_dict('records'))
            
            self.logger.info(f"Comprehensive evaluation completed for {task_type}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive evaluation: {e}")
            raise

    def _calculate_all_metrics(self, output: str, reference: str = None, 
                              task_type: str = "general", input_text: str = None,
                              inference_time: float = 0.0) -> Dict[str, float]:
        """Calculate all applicable metrics for the output"""
        metrics = {}
        
        try:
            # Basic content metrics
            metrics.update(self._calculate_basic_metrics(output))
            
            # Reference-based metrics (if reference available)
            if reference:
                metrics.update(self._calculate_reference_metrics(output, reference, task_type))
            
            # Input-based metrics (if input available)
            if input_text:
                metrics.update(self._calculate_input_based_metrics(output, input_text, task_type))
            
            # Advanced quality metrics
            metrics.update(self._calculate_advanced_metrics(output))
            
            # Performance metrics
            metrics.update(self._calculate_performance_metrics(output, inference_time))
            
            # Sentiment and style metrics
            metrics.update(self._calculate_sentiment_metrics(output))
            
            # Task-specific metrics
            metrics.update(self._calculate_task_specific_metrics(output, task_type, reference, input_text))
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
        
        return metrics

    def _calculate_basic_metrics(self, text: str) -> Dict[str, float]:
        """Calculate basic text metrics"""
        metrics = {}
        
        try:
            # Word and sentence counts
            words = text.split()
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            metrics["word_count"] = len(words)
            metrics["sentence_count"] = len(sentences)
            metrics["character_count"] = len(text)
            
            # Vocabulary diversity
            unique_words = set(word.lower() for word in words)
            metrics["vocabulary_diversity"] = len(unique_words) / len(words) if words else 0
            
            # Average sentence length
            metrics["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0
            
            # Readability score
            try:
                metrics["readability_score"] = textstat.flesch_reading_ease(text)
            except:
                metrics["readability_score"] = 0.0
            
        except Exception as e:
            self.logger.debug(f"Error in basic metrics: {e}")
        
        return metrics

    def _calculate_reference_metrics(self, output: str, reference: str, task_type: str) -> Dict[str, float]:
        """Calculate metrics that require a reference text"""
        metrics = {}
        
        try:
            # ROUGE scores
            rouge_scores = self.rouge_scorer.score(reference, output)
            metrics["rouge1"] = rouge_scores["rouge1"].fmeasure
            metrics["rouge2"] = rouge_scores["rouge2"].fmeasure
            metrics["rougeL"] = rouge_scores["rougeL"].fmeasure
            metrics["rougeLsum"] = rouge_scores["rougeLsum"].fmeasure
            
            # BERTScore
            if self.bertscore_available:
                try:
                    bert_results = self.bertscore.compute(
                        predictions=[output],
                        references=[reference],
                        lang="en"
                    )
                    metrics["bertscore_f1"] = bert_results["f1"][0]
                    metrics["bertscore_precision"] = bert_results["precision"][0]
                    metrics["bertscore_recall"] = bert_results["recall"][0]
                except Exception as e:
                    self.logger.debug(f"BERTScore calculation failed: {e}")
                    metrics["bertscore_f1"] = 0.0
                    metrics["bertscore_precision"] = 0.0
                    metrics["bertscore_recall"] = 0.0
            
            # Translation-specific metrics
            if task_type == "translation":
                metrics.update(self._calculate_translation_metrics(output, reference))
            
            # Coverage and factuality (reference-based)
            metrics["coverage_score"] = self._calculate_coverage(output, reference)
            
        except Exception as e:
            self.logger.debug(f"Error in reference metrics: {e}")
        
        return metrics

    def _calculate_translation_metrics(self, output: str, reference: str) -> Dict[str, float]:
        """Calculate translation-specific metrics"""
        metrics = {}
        
        try:
            # BLEU score
            reference_tokens = reference.split()
            output_tokens = output.split()
            
            try:
                metrics["bleu"] = sentence_bleu([reference_tokens], output_tokens)
                metrics["bleu_smooth"] = sentence_bleu(
                    [reference_tokens], output_tokens,
                    smoothing_function=self.bleu_smoother.method1
                )
            except Exception:
                metrics["bleu"] = 0.0
                metrics["bleu_smooth"] = 0.0
            
            # METEOR score
            try:
                metrics["meteor"] = meteor_score([reference_tokens], output_tokens)
            except Exception:
                metrics["meteor"] = 0.0
            
            # Character-level F-score (chrF)
            metrics["chrf"] = self._calculate_chrf(output, reference)
            
            # Length ratio
            metrics["length_ratio"] = len(output_tokens) / len(reference_tokens) if reference_tokens else 0
            
        except Exception as e:
            self.logger.debug(f"Error in translation metrics: {e}")
        
        return metrics

    def _calculate_chrf(self, hypothesis: str, reference: str) -> float:
        """Calculate character-level F-score"""
        try:
            # Simple character n-gram F-score implementation
            hyp_chars = list(hypothesis.replace(' ', ''))
            ref_chars = list(reference.replace(' ', ''))
            
            # Character 1-grams
            hyp_1grams = set(hyp_chars)
            ref_1grams = set(ref_chars)
            
            if not ref_1grams:
                return 0.0
            
            common_1grams = hyp_1grams.intersection(ref_1grams)
            
            precision = len(common_1grams) / len(hyp_1grams) if hyp_1grams else 0
            recall = len(common_1grams) / len(ref_1grams) if ref_1grams else 0
            
            if precision + recall == 0:
                return 0.0
            
            f_score = 2 * precision * recall / (precision + recall)
            return f_score
            
        except Exception:
            return 0.0

    def _calculate_input_based_metrics(self, output: str, input_text: str, task_type: str) -> Dict[str, float]:
        """Calculate metrics based on input text"""
        metrics = {}
        
        try:
            # Compression ratio for summarization
            if task_type == "summarization":
                input_words = len(input_text.split())
                output_words = len(output.split())
                metrics["compression_ratio"] = output_words / input_words if input_words > 0 else 0
            
            # Relevance score (simple word overlap)
            metrics["relevance_score"] = self._calculate_relevance(output, input_text)
            
        except Exception as e:
            self.logger.debug(f"Error in input-based metrics: {e}")
        
        return metrics

    def _calculate_advanced_metrics(self, text: str) -> Dict[str, float]:
        """Calculate advanced quality metrics"""
        metrics = {}
        
        try:
            # Coherence score
            metrics["coherence_score"] = self._calculate_coherence(text)
            
            # Fluency score
            metrics["fluency_score"] = self._calculate_fluency(text)
            
            # Perplexity
            if self.perplexity_available:
                metrics["perplexity"] = self._calculate_perplexity(text)
            else:
                metrics["perplexity"] = 0.0
            
            # Formality score
            metrics["formality_score"] = self._calculate_formality(text)
            
            # Novelty score
            metrics["novelty_score"] = self._calculate_novelty(text)
            
        except Exception as e:
            self.logger.debug(f"Error in advanced metrics: {e}")
        
        return metrics

    def _calculate_performance_metrics(self, text: str, inference_time: float) -> Dict[str, float]:
        """Calculate performance-related metrics"""
        metrics = {}
        
        try:
            word_count = len(text.split())
            
            # Tokens per second (approximate)
            if inference_time > 0:
                metrics["tokens_per_second"] = word_count / inference_time
                metrics["chars_per_second"] = len(text) / inference_time
            else:
                metrics["tokens_per_second"] = 0.0
                metrics["chars_per_second"] = 0.0
            
            # Throughput (requests per minute - hypothetical)
            if inference_time > 0:
                metrics["throughput"] = 60 / inference_time  # requests per minute
            else:
                metrics["throughput"] = 0.0
            
        except Exception as e:
            self.logger.debug(f"Error in performance metrics: {e}")
        
        return metrics

    def _calculate_sentiment_metrics(self, text: str) -> Dict[str, float]:
        """Calculate sentiment-related metrics"""
        metrics = {}
        
        try:
            if self.sentiment_available:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                metrics["sentiment_score"] = sentiment_scores["compound"]
                metrics["sentiment_positive"] = sentiment_scores["pos"]
                metrics["sentiment_negative"] = sentiment_scores["neg"]
                metrics["sentiment_neutral"] = sentiment_scores["neu"]
                metrics["sentiment_magnitude"] = abs(sentiment_scores["compound"])
            else:
                metrics["sentiment_score"] = 0.0
                metrics["sentiment_magnitude"] = 0.0
                
        except Exception as e:
            self.logger.debug(f"Error in sentiment metrics: {e}")
        
        return metrics

    def _calculate_task_specific_metrics(self, output: str, task_type: str, 
                                       reference: str = None, input_text: str = None) -> Dict[str, float]:
        """Calculate task-specific metrics"""
        metrics = {}
        
        try:
            if task_type == "summarization":
                metrics["factuality_score"] = self._estimate_factuality(output, input_text)
                
            elif task_type == "translation":
                metrics["adequacy_score"] = self._calculate_adequacy(output, reference)
                
            elif task_type == "chat":
                metrics["engagement_score"] = self._calculate_engagement(output)
                metrics["helpfulness_score"] = self._calculate_helpfulness(output)
            
        except Exception as e:
            self.logger.debug(f"Error in task-specific metrics: {e}")
        
        return metrics

    def _calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity using lightweight model"""
        try:
            if not text.strip():
                return float("inf")
            
            inputs = self.perplexity_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.perplexity_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            perplexity = torch.exp(loss).item()
            return min(perplexity, 1000.0)  # Cap at 1000
            
        except Exception:
            return float("inf")

    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score"""
        try:
            if not text.strip():
                return 0.0
            
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) < 2:
                return 0.7
            
            # Coherence indicators
            coherence_indicators = [
                'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                'consequently', 'meanwhile', 'subsequently', 'nevertheless',
                'thus', 'hence', 'accordingly', 'likewise', 'similarly'
            ]
            
            text_lower = text.lower()
            indicator_count = sum(1 for indicator in coherence_indicators if indicator in text_lower)
            
            base_score = 0.5
            indicator_score = min(0.3, indicator_count * 0.1)
            length_score = min(0.2, len(sentences) * 0.05)
            
            return min(1.0, base_score + indicator_score + length_score)
            
        except Exception:
            return 0.5

    def _calculate_fluency(self, text: str) -> float:
        """Calculate text fluency score"""
        try:
            if not text.strip():
                return 0.0
            
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            words = text.split()
            
            if not sentences or not words:
                return 0.0
            
            # Average sentence length (moderate is good)
            avg_sentence_length = len(words) / len(sentences)
            length_score = 1.0 - abs(avg_sentence_length - 15) / 15
            length_score = max(0.0, min(1.0, length_score))
            
            # Grammar indicators (basic)
            grammar_score = 0.7
            
            if text.count(',,') > 0 or text.count('..') > text.count('...'):
                grammar_score -= 0.2
            
            if text and text[0].islower():
                grammar_score -= 0.1
            
            return (length_score + grammar_score) / 2
            
        except Exception:
            return 0.5

    def _calculate_coverage(self, output: str, reference: str) -> float:
        """Calculate content coverage"""
        try:
            if not output or not reference:
                return 0.0
            
            output_words = set(output.lower().split())
            reference_words = set(reference.lower().split())
            
            if not reference_words:
                return 0.0
            
            covered_words = output_words.intersection(reference_words)
            return len(covered_words) / len(reference_words)
            
        except Exception:
            return 0.0

    def _calculate_relevance(self, output: str, input_text: str) -> float:
        """Calculate relevance to input"""
        try:
            if not output or not input_text:
                return 0.5
            
            output_words = set(output.lower().split())
            input_words = set(input_text.lower().split())
            
            if not input_words:
                return 0.5
            
            overlap = len(output_words.intersection(input_words))
            max_words = max(len(output_words), len(input_words))
            
            return overlap / max_words if max_words > 0 else 0.0
            
        except Exception:
            return 0.5

    def _calculate_formality(self, text: str) -> float:
        """Calculate formality score"""
        try:
            formal_indicators = [
                'furthermore', 'moreover', 'however', 'therefore', 'consequently',
                'nevertheless', 'accordingly', 'subsequently', 'establish',
                'demonstrate', 'indicate', 'suggest', 'conclude'
            ]
            
            informal_indicators = [
                'gonna', 'wanna', 'kinda', 'sorta', 'yeah', 'ok', 'hey',
                'awesome', 'cool', 'super', 'totally', 'really'
            ]
            
            text_lower = text.lower()
            words = text_lower.split()
            
            formal_count = sum(1 for word in words if word in formal_indicators)
            informal_count = sum(1 for word in words if word in informal_indicators)
            
            total_indicators = formal_count + informal_count
            if total_indicators == 0:
                return 0.5  # Neutral
            
            return formal_count / total_indicators
            
        except Exception:
            return 0.5

    def _calculate_novelty(self, text: str) -> float:
        """Calculate novelty/creativity score"""
        try:
            # Simple approach: measure vocabulary diversity and uncommon words
            words = text.lower().split()
            unique_words = set(words)
            
            if not words:
                return 0.0
            
            # Basic diversity
            diversity = len(unique_words) / len(words)
            
            # Uncommon word bonus (words longer than 7 characters)
            long_words = [w for w in unique_words if len(w) > 7]
            long_word_ratio = len(long_words) / len(unique_words) if unique_words else 0
            
            novelty = (diversity * 0.7) + (long_word_ratio * 0.3)
            return min(1.0, novelty)
            
        except Exception:
            return 0.3

    def _estimate_factuality(self, output: str, input_text: str = None) -> float:
        """Estimate factual accuracy (heuristic-based)"""
        try:
            # Look for factual indicators vs uncertain language
            certain_indicators = [
                'according to', 'research shows', 'studies indicate',
                'data reveals', 'statistics show', 'evidence suggests'
            ]
            
            uncertain_indicators = [
                'might', 'maybe', 'possibly', 'perhaps', 'could be',
                'seems like', 'appears to', 'suggests that'
            ]
            
            text_lower = output.lower()
            
            certain_count = sum(1 for indicator in certain_indicators if indicator in text_lower)
            uncertain_count = sum(1 for indicator in uncertain_indicators if indicator in text_lower)
            
            # Base score
            base_score = 0.6
            
            # Adjust based on certainty indicators
            if certain_count > uncertain_count:
                return min(1.0, base_score + 0.2)
            elif uncertain_count > certain_count:
                return max(0.0, base_score - 0.2)
            
            return base_score
            
        except Exception:
            return 0.6

    def _calculate_adequacy(self, output: str, reference: str) -> float:
        """Calculate translation adequacy"""
        return self._calculate_coverage(output, reference)

    def _calculate_engagement(self, text: str) -> float:
        """Calculate engagement score for chat responses"""
        try:
            engagement_indicators = [
                '?', '!', 'you', 'your', 'what', 'how', 'why', 'when', 'where',
                'think', 'feel', 'believe', 'opinion', 'interesting', 'amazing'
            ]
            
            text_lower = text.lower()
            indicator_count = sum(1 for indicator in engagement_indicators if indicator in text_lower)
            
            words = text.split()
            engagement_ratio = indicator_count / max(len(words), 1)
            
            return min(1.0, engagement_ratio * 10)
            
        except Exception:
            return 0.5

    def _calculate_helpfulness(self, text: str) -> float:
        """Calculate helpfulness score"""
        try:
            helpful_indicators = [
                'help', 'assist', 'support', 'guide', 'explain', 'clarify',
                'solution', 'answer', 'resolve', 'suggest', 'recommend'
            ]
            
            text_lower = text.lower()
            helpful_count = sum(1 for indicator in helpful_indicators if indicator in text_lower)
            
            words = text.split()
            helpfulness_ratio = helpful_count / max(len(words), 1)
            
            return min(1.0, helpfulness_ratio * 15)
            
        except Exception:
            return 0.5

    def _get_default_metrics(self) -> Dict[str, float]:
        """Get default metric values for failed outputs"""
        return {metric: 0.0 for metric in self.metric_definitions.keys()}

    def _assess_overall_quality(self, metrics: Dict[str, float], task_type: str) -> str:
        """Assess overall quality category"""
        try:
            relevant_metrics = []
            
            # Select key metrics based on task type
            if task_type == "summarization":
                key_metrics = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "coherence_score"]
            elif task_type == "translation":
                key_metrics = ["bleu", "meteor", "bertscore_f1", "fluency_score"]
            elif task_type == "chat":
                key_metrics = ["coherence_score", "fluency_score", "engagement_score", "relevance_score"]
            else:
                key_metrics = ["coherence_score", "fluency_score", "relevance_score"]
            
            # Calculate average quality score
            for metric in key_metrics:
                if metric in metrics and metric in self.metric_definitions:
                    score = metrics[metric]
                    threshold = self.metric_definitions[metric]["good_threshold"]
                    
                    # Normalize based on threshold
                    if "lower is better" in self.metric_definitions[metric]["range"]:
                        normalized = 1.0 / (1.0 + score / threshold)
                    else:
                        normalized = score / threshold
                    
                    relevant_metrics.append(min(1.0, normalized))
            
            if not relevant_metrics:
                return "unknown"
            
            avg_quality = sum(relevant_metrics) / len(relevant_metrics)
            
            if avg_quality >= 0.8:
                return "excellent"
            elif avg_quality >= 0.6:
                return "good"
            elif avg_quality >= 0.4:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "unknown"

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

    def get_available_metrics(self, task_type: str = None) -> List[Dict[str, str]]:
        """Get list of available metrics, optionally filtered by task type"""
        available = []
        
        for metric_id, metric_info in self.metric_definitions.items():
            if task_type is None or task_type in metric_info.get("task_types", ["all"]) or "all" in metric_info.get("task_types", []):
                available.append({
                    "id": metric_id,
                    "name": metric_info["name"],
                    "description": metric_info["description"],
                    "category": metric_info["category"]
                })
        
        return available

    def explain_metric(self, metric_id: str) -> Dict[str, Any]:
        """Get detailed explanation of a specific metric"""
        if metric_id in self.metric_definitions:
            return self.metric_definitions[metric_id]
        else:
            return {"error": f"Metric {metric_id} not found"}

    def compare_models(self, evaluation_df: pd.DataFrame, metrics: List[str] = None) -> Dict[str, Any]:
        """Compare models across specified metrics"""
        try:
            if evaluation_df.empty:
                return {"error": "No evaluation data provided"}
            
            if metrics is None:
                metrics = [col for col in evaluation_df.columns 
                          if col not in ["model", "inference_time", "success", "quality_category"]]
            
            comparison = {
                "model_count": len(evaluation_df),
                "metric_comparisons": {},
                "rankings": {},
                "summary": {}
            }
            
            for metric in metrics:
                if metric in evaluation_df.columns:
                    metric_data = evaluation_df[metric].dropna()
                    
                    if len(metric_data) > 0:
                        # Basic statistics
                        comparison["metric_comparisons"][metric] = {
                            "mean": float(metric_data.mean()),
                            "std": float(metric_data.std()),
                            "min": float(metric_data.min()),
                            "max": float(metric_data.max()),
                            "best_model": evaluation_df.loc[metric_data.idxmax(), "model"],
                            "worst_model": evaluation_df.loc[metric_data.idxmin(), "model"]
                        }
                        
                        # Rankings
                        sorted_models = evaluation_df.nlargest(len(evaluation_df), metric)
                        comparison["rankings"][metric] = [
                            {"model": row["model"], "score": row[metric], "rank": i+1}
                            for i, (_, row) in enumerate(sorted_models.iterrows())
                        ]
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            return {"error": str(e)}

    def export_evaluation(self, evaluation_df: pd.DataFrame, format: str = "json") -> str:
        """Export evaluation results in specified format"""
        try:
            if format.lower() == "json":
                return evaluation_df.to_json(orient="records", indent=2)
            elif format.lower() == "csv":
                return evaluation_df.to_csv(index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting evaluation: {e}")
            return f"Export error: {str(e)}"