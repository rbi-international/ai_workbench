from utils.logger import setup_logger
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
import difflib
from collections import Counter
import re

class ModelFusion:
    """
    Advanced model fusion system that combines outputs from multiple AI models
    using various strategies for improved performance
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Fusion strategies
        self.fusion_strategies = {
            "weighted_average": self._weighted_average_fusion,
            "best_model": self._best_model_fusion,
            "ensemble_voting": self._ensemble_voting_fusion,
            "length_weighted": self._length_weighted_fusion,
            "quality_weighted": self._quality_weighted_fusion,
            "consensus": self._consensus_fusion,
            "hybrid": self._hybrid_fusion
        }
        
        self.logger.info("Model fusion system initialized with multiple strategies")

    def fuse_outputs(self, results: List[Dict], reference: str = None, strategy: str = "weighted_average", task_type: str = "general") -> Dict[str, Any]:
        """
        Fuse outputs from multiple models using specified strategy
        
        Args:
            results: List of model result dictionaries
            reference: Optional reference text for quality assessment
            strategy: Fusion strategy to use
            task_type: Type of task (summarization, translation, chat)
            
        Returns:
            Dictionary with fused output and metadata
        """
        try:
            # Filter successful results
            successful_results = [r for r in results if r.get("output") and r.get("success", True)]
            
            if not successful_results:
                self.logger.warning("No successful results to fuse")
                return {
                    "fused_output": None,
                    "strategy": strategy,
                    "weights": {},
                    "confidence": 0.0,
                    "error": "No successful results available"
                }
            
            if len(successful_results) == 1:
                # Only one model succeeded
                result = successful_results[0]
                return {
                    "fused_output": result["output"],
                    "strategy": "single_model",
                    "weights": {result["model"]: 1.0},
                    "confidence": 0.8,
                    "metadata": {
                        "single_model_used": result["model"],
                        "reason": "Only one successful model"
                    }
                }
            
            # Apply fusion strategy
            if strategy not in self.fusion_strategies:
                self.logger.warning(f"Unknown strategy {strategy}, using weighted_average")
                strategy = "weighted_average"
            
            fusion_func = self.fusion_strategies[strategy]
            fusion_result = fusion_func(successful_results, reference, task_type)
            
            # Add metadata
            fusion_result.update({
                "strategy": strategy,
                "num_models": len(successful_results),
                "model_names": [r["model"] for r in successful_results],
                "task_type": task_type
            })
            
            self.logger.info(f"Fusion completed using {strategy} strategy with {len(successful_results)} models")
            return fusion_result
            
        except Exception as e:
            self.logger.error(f"Error in model fusion: {e}")
            return {
                "fused_output": None,
                "strategy": strategy,
                "weights": {},
                "confidence": 0.0,
                "error": str(e)
            }

    def _weighted_average_fusion(self, results: List[Dict], reference: str, task_type: str) -> Dict[str, Any]:
        """Fusion based on model performance weights"""
        outputs = [r["output"] for r in results]
        models = [r["model"] for r in results]
        
        # Calculate weights based on various factors
        weights = self._calculate_performance_weights(results, reference)
        
        # For text, we can't really "average" - use weighted selection or combination
        if task_type in ["summarization", "translation"]:
            fused_output = self._combine_text_weighted(outputs, weights, models)
        else:
            # For chat, select best model based on weights
            best_idx = np.argmax(list(weights.values()))
            fused_output = outputs[best_idx]
        
        confidence = self._calculate_fusion_confidence(results, weights)
        
        return {
            "fused_output": fused_output,
            "weights": weights,
            "confidence": confidence,
            "method": "weighted_combination"
        }

    def _best_model_fusion(self, results: List[Dict], reference: str, task_type: str) -> Dict[str, Any]:
        """Select output from the best performing model"""
        # Calculate quality scores for each model
        quality_scores = {}
        
        for result in results:
            model = result["model"]
            output = result["output"]
            
            # Calculate quality based on multiple factors
            quality = 0.0
            
            # Speed factor (faster is better, but not too much weight)
            inference_time = result.get("inference_time", 1.0)
            speed_score = 1.0 / (1.0 + inference_time / 10.0)  # Normalize
            quality += speed_score * 0.2
            
            # Length factor (reasonable length)
            length = len(output.split())
            if task_type == "summarization":
                # Prefer moderate length summaries
                length_score = 1.0 - abs(length - 50) / 100.0
            else:
                # For other tasks, moderate preference for longer outputs
                length_score = min(1.0, length / 100.0)
            
            quality += max(0, length_score) * 0.3
            
            # Reference comparison if available
            if reference:
                rouge_score = self.rouge_scorer.score(reference, output)["rougeL"].fmeasure
                quality += rouge_score * 0.5
            else:
                # Without reference, use content quality heuristics
                content_quality = self._assess_content_quality(output, task_type)
                quality += content_quality * 0.5
            
            quality_scores[model] = quality
        
        # Select best model
        best_model = max(quality_scores, key=quality_scores.get)
        best_output = next(r["output"] for r in results if r["model"] == best_model)
        
        # Create weights (winner takes most, others get small weights)
        weights = {}
        total_quality = sum(quality_scores.values())
        for model, score in quality_scores.items():
            if model == best_model:
                weights[model] = 0.7 + (score / total_quality) * 0.3
            else:
                weights[model] = (score / total_quality) * 0.3
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {k: v / weight_sum for k, v in weights.items()}
        
        return {
            "fused_output": best_output,
            "weights": weights,
            "confidence": quality_scores[best_model],
            "method": "best_model_selection",
            "quality_scores": quality_scores,
            "selected_model": best_model
        }

    def _ensemble_voting_fusion(self, results: List[Dict], reference: str, task_type: str) -> Dict[str, Any]:
        """Combine outputs using ensemble voting techniques"""
        outputs = [r["output"] for r in results]
        models = [r["model"] for r in results]
        
        if task_type == "chat":
            # For chat, find most similar responses and combine
            return self._consensus_fusion(results, reference, task_type)
        
        # For summarization/translation, extract key elements and vote
        if task_type == "summarization":
            fused_output = self._vote_summarization(outputs, models)
        elif task_type == "translation":
            fused_output = self._vote_translation(outputs, models)
        else:
            fused_output = self._vote_generic(outputs, models)
        
        # Equal weights for voting
        weights = {model: 1.0 / len(models) for model in models}
        confidence = self._calculate_consensus_confidence(outputs)
        
        return {
            "fused_output": fused_output,
            "weights": weights,
            "confidence": confidence,
            "method": "ensemble_voting"
        }

    def _consensus_fusion(self, results: List[Dict], reference: str, task_type: str) -> Dict[str, Any]:
        """Find consensus among model outputs"""
        outputs = [r["output"] for r in results]
        models = [r["model"] for r in results]
        
        # Calculate pairwise similarities
        similarity_matrix = self._calculate_similarity_matrix(outputs)
        
        # Find the output with highest average similarity to others
        avg_similarities = np.mean(similarity_matrix, axis=1)
        consensus_idx = np.argmax(avg_similarities)
        
        # Use the most consensual output as base
        base_output = outputs[consensus_idx]
        
        # Calculate confidence based on consensus level
        consensus_score = avg_similarities[consensus_idx]
        
        # Weight models based on similarity to consensus
        weights = {}
        for i, model in enumerate(models):
            weights[model] = similarity_matrix[consensus_idx][i]
        
        # Normalize weights
        weight_sum = sum(weights.values())
        if weight_sum > 0:
            weights = {k: v / weight_sum for k, v in weights.items()}
        else:
            weights = {model: 1.0 / len(models) for model in models}
        
        return {
            "fused_output": base_output,
            "weights": weights,
            "confidence": consensus_score,
            "method": "consensus_selection",
            "consensus_model": models[consensus_idx],
            "similarity_matrix": similarity_matrix.tolist()
        }

    def _hybrid_fusion(self, results: List[Dict], reference: str, task_type: str) -> Dict[str, Any]:
        """Hybrid approach combining multiple fusion strategies"""
        # Apply multiple strategies
        strategies = ["weighted_average", "best_model", "consensus"]
        strategy_results = {}
        
        for strategy in strategies:
            try:
                strategy_func = self.fusion_strategies[strategy]
                result = strategy_func(results, reference, task_type)
                strategy_results[strategy] = result
            except Exception as e:
                self.logger.warning(f"Strategy {strategy} failed: {e}")
                continue
        
        if not strategy_results:
            # Fallback to simple selection
            return self._best_model_fusion(results, reference, task_type)
        
        # Select best strategy result based on confidence
        best_strategy = max(strategy_results.keys(), 
                           key=lambda s: strategy_results[s].get("confidence", 0))
        
        best_result = strategy_results[best_strategy]
        
        # Add hybrid metadata
        best_result.update({
            "method": "hybrid_fusion",
            "selected_strategy": best_strategy,
            "strategy_confidences": {s: r.get("confidence", 0) for s, r in strategy_results.items()}
        })
        
        return best_result

    def _calculate_performance_weights(self, results: List[Dict], reference: str) -> Dict[str, float]:
        """Calculate performance-based weights for models"""
        weights = {}
        
        for result in results:
            model = result["model"]
            output = result["output"]
            
            weight = 0.0
            
            # Quality factor
            if reference:
                rouge_score = self.rouge_scorer.score(reference, output)["rougeL"].fmeasure
                weight += rouge_score * 0.6
            else:
                content_quality = self._assess_content_quality(output, "general")
                weight += content_quality * 0.6
            
            # Speed factor (inverse of time, normalized)
            inference_time = result.get("inference_time", 1.0)
            speed_factor = 1.0 / (1.0 + inference_time)
            weight += speed_factor * 0.2
            
            # Length appropriateness
            length = len(output.split())
            length_factor = min(1.0, length / 50.0) if length < 50 else max(0.5, 100.0 / length)
            weight += length_factor * 0.2
            
            weights[model] = max(0.1, weight)  # Ensure minimum weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights fallback
            weights = {model: 1.0 / len(results) for model in [r["model"] for r in results]}
        
        return weights

    def _assess_content_quality(self, text: str, task_type: str) -> float:
        """Assess content quality using heuristics"""
        if not text or not text.strip():
            return 0.0
        
        quality_score = 0.0
        
        # Basic structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 0:
            quality_score += 0.3
        
        # Vocabulary diversity
        words = text.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            diversity = len(unique_words) / len(words)
            quality_score += diversity * 0.3
        
        # No obvious errors
        if not re.search(r'\b(error|failed|invalid)\b', text.lower()):
            quality_score += 0.2
        
        # Task-specific quality
        if task_type == "summarization":
            # Check for summary indicators
            if any(word in text.lower() for word in ['summary', 'overview', 'key', 'main', 'important']):
                quality_score += 0.2
        elif task_type == "translation":
            # Check for completeness
            if len(text) > 10 and not text.endswith('...'):
                quality_score += 0.2
        
        return min(1.0, quality_score)

    def _combine_text_weighted(self, outputs: List[str], weights: Dict[str, float], models: List[str]) -> str:
        """Combine text outputs using weights"""
        # For text combination, we'll select the best parts from each output
        # This is a simplified approach - more sophisticated methods could be used
        
        # Find the output with highest weight as base
        model_weights = [weights.get(model, 0) for model in models]
        best_idx = np.argmax(model_weights)
        base_output = outputs[best_idx]
        
        # For now, return the best weighted output
        # TODO: Implement sentence-level fusion for better results
        return base_output

    def _calculate_similarity_matrix(self, outputs: List[str]) -> np.ndarray:
        """Calculate pairwise similarity matrix for outputs"""
        n = len(outputs)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Use multiple similarity measures
                    rouge_sim = self.rouge_scorer.score(outputs[i], outputs[j])["rougeL"].fmeasure
                    
                    # Word overlap similarity
                    words_i = set(outputs[i].lower().split())
                    words_j = set(outputs[j].lower().split())
                    if len(words_i.union(words_j)) > 0:
                        word_sim = len(words_i.intersection(words_j)) / len(words_i.union(words_j))
                    else:
                        word_sim = 0.0
                    
                    # Sequence similarity
                    seq_sim = difflib.SequenceMatcher(None, outputs[i], outputs[j]).ratio()
                    
                    # Combined similarity
                    similarity = (rouge_sim * 0.5 + word_sim * 0.3 + seq_sim * 0.2)
                    similarity_matrix[i][j] = similarity
        
        return similarity_matrix

    def _calculate_fusion_confidence(self, results: List[Dict], weights: Dict[str, float]) -> float:
        """Calculate confidence in fusion result"""
        # Base confidence on model agreement and individual confidences
        
        # Weight distribution factor (more even = less confident)
        weight_values = list(weights.values())
        weight_entropy = -sum(w * np.log(w + 1e-10) for w in weight_values)
        max_entropy = np.log(len(weight_values))
        entropy_factor = 1.0 - (weight_entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Speed consistency factor
        times = [r.get("inference_time", 1.0) for r in results]
        time_std = np.std(times)
        time_factor = 1.0 / (1.0 + time_std)
        
        # Quality factors
        quality_issues = []
        for result in results:
            issues = result.get("quality_issues", [])
            quality_issues.extend(issues)
        
        quality_factor = max(0.0, 1.0 - len(quality_issues) / (len(results) * 3))
        
        # Combined confidence
        confidence = (entropy_factor * 0.4 + time_factor * 0.3 + quality_factor * 0.3)
        
        return min(1.0, max(0.0, confidence))

    def _calculate_consensus_confidence(self, outputs: List[str]) -> float:
        """Calculate confidence based on consensus among outputs"""
        if len(outputs) < 2:
            return 0.8
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = self.rouge_scorer.score(outputs[i], outputs[j])["rougeL"].fmeasure
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # High consensus = high confidence
        return avg_similarity

    def _vote_summarization(self, outputs: List[str], models: List[str]) -> str:
        """Voting-based fusion for summarization"""
        # Extract key sentences from all summaries
        all_sentences = []
        for output in outputs:
            sentences = [s.strip() for s in output.split('.') if s.strip()]
            all_sentences.extend(sentences)
        
        # Score sentences by frequency and quality
        sentence_scores = {}
        for sentence in all_sentences:
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Frequency score
            freq_score = all_sentences.count(sentence)
            
            # Length score (prefer moderate length)
            length_score = min(1.0, len(sentence.split()) / 20.0)
            
            # Combined score
            sentence_scores[sentence] = freq_score + length_score
        
        # Select top sentences
        if sentence_scores:
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            # Take top 3-5 sentences or until we reach ~100 words
            selected_sentences = []
            word_count = 0
            
            for sentence, score in top_sentences:
                if word_count > 80:  # Target length
                    break
                selected_sentences.append(sentence)
                word_count += len(sentence.split())
            
            return '. '.join(selected_sentences) + '.'
        else:
            # Fallback to first output
            return outputs[0] if outputs else ""

    def _vote_translation(self, outputs: List[str], models: List[str]) -> str:
        """Voting-based fusion for translation"""
        # For translation, find most common words/phrases
        all_words = []
        for output in outputs:
            words = output.split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Build translation by selecting most frequent words
        # This is a simplified approach - real translation fusion is more complex
        
        # For now, return the longest translation (often more complete)
        return max(outputs, key=len) if outputs else ""

    def _vote_generic(self, outputs: List[str], models: List[str]) -> str:
        """Generic voting fusion"""
        # Simple approach: return the output with median length
        if not outputs:
            return ""
        
        lengths = [len(output.split()) for output in outputs]
        median_length = np.median(lengths)
        
        # Find output closest to median length
        closest_idx = min(range(len(lengths)), key=lambda i: abs(lengths[i] - median_length))
        
        return outputs[closest_idx]

    def _length_weighted_fusion(self, results: List[Dict], reference: str, task_type: str) -> Dict[str, Any]:
        """Fusion based on output length appropriateness"""
        outputs = [r["output"] for r in results]
        models = [r["model"] for r in results]
        
        # Calculate ideal length based on task
        if task_type == "summarization":
            ideal_length = 50  # words
        elif task_type == "translation":
            # Base on input length if available
            ideal_length = 100  # words
        else:
            ideal_length = 75  # words
        
        # Calculate weights based on length appropriateness
        weights = {}
        for i, result in enumerate(results):
            model = result["model"]
            output_length = len(result["output"].split())
            
            # Distance from ideal length
            length_diff = abs(output_length - ideal_length)
            
            # Convert to weight (closer to ideal = higher weight)
            weight = 1.0 / (1.0 + length_diff / ideal_length)
            weights[model] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Select output with highest weight
        best_model = max(weights, key=weights.get)
        fused_output = next(r["output"] for r in results if r["model"] == best_model)
        
        return {
            "fused_output": fused_output,
            "weights": weights,
            "confidence": weights[best_model],
            "method": "length_weighted_selection",
            "ideal_length": ideal_length,
            "selected_model": best_model
        }

    def _quality_weighted_fusion(self, results: List[Dict], reference: str, task_type: str) -> Dict[str, Any]:
        """Fusion based on quality assessment"""
        weights = {}
        
        for result in results:
            model = result["model"]
            output = result["output"]
            
            # Calculate quality score
            quality = 0.0
            
            # Content quality
            content_quality = self._assess_content_quality(output, task_type)
            quality += content_quality * 0.4
            
            # Reference similarity if available
            if reference:
                rouge_score = self.rouge_scorer.score(reference, output)["rougeL"].fmeasure
                quality += rouge_score * 0.4
            
            # Quality issues penalty
            quality_issues = result.get("quality_issues", [])
            quality_penalty = len(quality_issues) * 0.1
            quality = max(0.0, quality - quality_penalty)
            
            # Speed bonus (small)
            inference_time = result.get("inference_time", 1.0)
            speed_bonus = 0.2 / (1.0 + inference_time)
            quality += speed_bonus
            
            weights[model] = quality
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {result["model"]: 1.0 / len(results) for result in results}
        
        # Select best quality output
        best_model = max(weights, key=weights.get)
        fused_output = next(r["output"] for r in results if r["model"] == best_model)
        
        confidence = weights[best_model]
        
        return {
            "fused_output": fused_output,
            "weights": weights,
            "confidence": confidence,
            "method": "quality_weighted_selection",
            "selected_model": best_model
        }

    def get_available_strategies(self) -> Dict[str, str]:
        """Get available fusion strategies with descriptions"""
        return {
            "weighted_average": "Combines outputs based on model performance weights",
            "best_model": "Selects output from the best performing model",
            "ensemble_voting": "Uses voting mechanisms to combine model outputs",
            "length_weighted": "Weights models based on output length appropriateness",
            "quality_weighted": "Weights models based on assessed output quality",
            "consensus": "Finds consensus among similar model outputs",
            "hybrid": "Combines multiple fusion strategies for best results"
        }

    def analyze_fusion_effectiveness(self, results: List[Dict], fused_result: Dict, reference: str = None) -> Dict[str, Any]:
        """
        Analyze how effective the fusion was compared to individual models
        
        Args:
            results: Original model results
            fused_result: Result from fusion
            reference: Optional reference text
            
        Returns:
            Analysis of fusion effectiveness
        """
        try:
            analysis = {
                "fusion_strategy": fused_result.get("strategy", "unknown"),
                "improvement_over_best": 0.0,
                "improvement_over_average": 0.0,
                "confidence": fused_result.get("confidence", 0.0),
                "individual_qualities": {},
                "fusion_quality": 0.0
            }
            
            fused_output = fused_result.get("fused_output", "")
            
            # Calculate quality scores for individual models
            individual_scores = []
            for result in results:
                if result.get("output"):
                    if reference:
                        score = self.rouge_scorer.score(reference, result["output"])["rougeL"].fmeasure
                    else:
                        score = self._assess_content_quality(result["output"], "general")
                    
                    analysis["individual_qualities"][result["model"]] = score
                    individual_scores.append(score)
            
            # Calculate fusion quality
            if fused_output:
                if reference:
                    fusion_score = self.rouge_scorer.score(reference, fused_output)["rougeL"].fmeasure
                else:
                    fusion_score = self._assess_content_quality(fused_output, "general")
                
                analysis["fusion_quality"] = fusion_score
                
                # Compare to individual models
                if individual_scores:
                    best_individual = max(individual_scores)
                    avg_individual = np.mean(individual_scores)
                    
                    analysis["improvement_over_best"] = fusion_score - best_individual
                    analysis["improvement_over_average"] = fusion_score - avg_individual
            
            # Additional insights
            analysis["insights"] = self._generate_fusion_insights(analysis, fused_result)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing fusion effectiveness: {e}")
            return {"error": str(e)}

    def _generate_fusion_insights(self, analysis: Dict, fused_result: Dict) -> List[str]:
        """Generate insights about fusion performance"""
        insights = []
        
        strategy = analysis.get("fusion_strategy", "unknown")
        improvement_best = analysis.get("improvement_over_best", 0.0)
        improvement_avg = analysis.get("improvement_over_average", 0.0)
        confidence = analysis.get("confidence", 0.0)
        
        # Strategy effectiveness
        if improvement_best > 0.05:
            insights.append(f"✅ {strategy} fusion improved over best individual model by {improvement_best:.3f}")
        elif improvement_best > -0.02:
            insights.append(f"➡️ {strategy} fusion performed similarly to best individual model")
        else:
            insights.append(f"❌ {strategy} fusion underperformed compared to best individual model")
        
        # Confidence assessment
        if confidence > 0.8:
            insights.append("🎯 High confidence in fusion result")
        elif confidence > 0.6:
            insights.append("⚖️ Moderate confidence in fusion result")
        else:
            insights.append("⚠️ Low confidence in fusion result")
        
        # Strategy-specific insights
        if strategy == "consensus":
            weights = fused_result.get("weights", {})
            max_weight = max(weights.values()) if weights else 0
            if max_weight > 0.8:
                insights.append("🤝 Strong consensus among models")
            else:
                insights.append("🤔 Limited consensus among models")
        
        elif strategy == "hybrid":
            selected_strategy = fused_result.get("selected_strategy", "unknown")
            insights.append(f"🔀 Hybrid approach selected {selected_strategy} as best strategy")
        
        # Model weight distribution
        weights = fused_result.get("weights", {})
        if weights:
            dominant_model = max(weights, key=weights.get)
            dominant_weight = weights[dominant_model]
            
            if dominant_weight > 0.7:
                insights.append(f"👑 {dominant_model} dominated the fusion ({dominant_weight:.2f} weight)")
            else:
                insights.append("⚖️ Weights were distributed across multiple models")
        
        return insights

    def recommend_fusion_strategy(self, results: List[Dict], task_type: str, reference: str = None) -> Dict[str, Any]:
        """
        Recommend the best fusion strategy for given results
        
        Args:
            results: Model results
            task_type: Type of task
            reference: Optional reference text
            
        Returns:
            Recommendation with reasoning
        """
        try:
            successful_results = [r for r in results if r.get("output") and r.get("success", True)]
            
            if len(successful_results) <= 1:
                return {
                    "recommended_strategy": "single_model",
                    "reason": "Only one successful model available",
                    "confidence": 0.9
                }
            
            # Analyze model characteristics
            model_analysis = self._analyze_model_characteristics(successful_results, reference)
            
            # Decision logic
            recommendation = self._decide_fusion_strategy(model_analysis, task_type)
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Error recommending fusion strategy: {e}")
            return {
                "recommended_strategy": "weighted_average",
                "reason": f"Error in analysis: {str(e)}",
                "confidence": 0.5
            }

    def _analyze_model_characteristics(self, results: List[Dict], reference: str) -> Dict[str, Any]:
        """Analyze characteristics of model results"""
        analysis = {
            "quality_variance": 0.0,
            "speed_variance": 0.0,
            "length_variance": 0.0,
            "consensus_level": 0.0,
            "has_reference": reference is not None,
            "num_models": len(results)
        }
        
        # Quality analysis
        if reference:
            qualities = []
            for result in results:
                score = self.rouge_scorer.score(reference, result["output"])["rougeL"].fmeasure
                qualities.append(score)
            analysis["quality_variance"] = np.var(qualities) if len(qualities) > 1 else 0.0
        
        # Speed analysis
        speeds = [result.get("inference_time", 1.0) for result in results]
        analysis["speed_variance"] = np.var(speeds) if len(speeds) > 1 else 0.0
        
        # Length analysis
        lengths = [len(result["output"].split()) for result in results]
        analysis["length_variance"] = np.var(lengths) if len(lengths) > 1 else 0.0
        
        # Consensus analysis
        outputs = [result["output"] for result in results]
        similarity_matrix = self._calculate_similarity_matrix(outputs)
        analysis["consensus_level"] = np.mean(similarity_matrix)
        
        return analysis

    def _decide_fusion_strategy(self, analysis: Dict, task_type: str) -> Dict[str, Any]:
        """Decide on fusion strategy based on analysis"""
        
        quality_var = analysis["quality_variance"]
        consensus = analysis["consensus_level"]
        has_reference = analysis["has_reference"]
        num_models = analysis["num_models"]
        
        # Decision tree
        if consensus > 0.8:
            return {
                "recommended_strategy": "consensus",
                "reason": "High consensus among models suggests reliable agreement",
                "confidence": 0.9
            }
        
        elif quality_var > 0.1 and has_reference:
            return {
                "recommended_strategy": "quality_weighted",
                "reason": "High quality variance with reference available for accurate weighting",
                "confidence": 0.85
            }
        
        elif num_models >= 3:
            return {
                "recommended_strategy": "ensemble_voting",
                "reason": "Multiple models available for ensemble voting approach",
                "confidence": 0.8
            }
        
        elif has_reference:
            return {
                "recommended_strategy": "weighted_average",
                "reason": "Reference available for performance-based weighting",
                "confidence": 0.75
            }
        
        else:
            return {
                "recommended_strategy": "best_model",
                "reason": "No reference available, select best model based on heuristics",
                "confidence": 0.7
            }