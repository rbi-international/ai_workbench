from utils.logger import setup_logger
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np

class AITutor:
    """
    Enhanced AI Tutor that provides detailed explanations about model performance,
    recommendations for improvement, and educational insights
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        # Knowledge base for explanations
        self.metric_explanations = {
            "rouge1": {
                "description": "Measures overlap of single words (unigrams) between generated and reference text",
                "range": "0-1 (higher is better)",
                "good_score": 0.4,
                "interpretation": {
                    "high": "Strong word-level similarity to reference",
                    "medium": "Moderate word overlap",
                    "low": "Poor word-level match"
                }
            },
            "rouge2": {
                "description": "Measures overlap of two-word sequences (bigrams) between texts",
                "range": "0-1 (higher is better)", 
                "good_score": 0.2,
                "interpretation": {
                    "high": "Good phrase-level similarity",
                    "medium": "Some phrase overlap",
                    "low": "Little phrase-level match"
                }
            },
            "rougeL": {
                "description": "Measures longest common subsequence between texts",
                "range": "0-1 (higher is better)",
                "good_score": 0.3,
                "interpretation": {
                    "high": "Strong structural similarity",
                    "medium": "Moderate structural match",
                    "low": "Poor structural alignment"
                }
            },
            "bertscore": {
                "description": "Uses BERT embeddings to measure semantic similarity",
                "range": "0-1 (higher is better)",
                "good_score": 0.5,
                "interpretation": {
                    "high": "Strong semantic similarity",
                    "medium": "Moderate semantic match", 
                    "low": "Poor semantic alignment"
                }
            },
            "bleu": {
                "description": "Measures n-gram precision between translation and reference",
                "range": "0-1 (higher is better)",
                "good_score": 0.3,
                "interpretation": {
                    "high": "Excellent translation quality",
                    "medium": "Good translation quality",
                    "low": "Poor translation quality"
                }
            },
            "meteor": {
                "description": "Considers word order, synonyms, and stemming in translation evaluation",
                "range": "0-1 (higher is better)",
                "good_score": 0.4,
                "interpretation": {
                    "high": "High translation fluency",
                    "medium": "Moderate fluency",
                    "low": "Low fluency"
                }
            },
            "perplexity": {
                "description": "Measures how well a model predicts the text (lower is better)",
                "range": "1+ (lower is better)",
                "good_score": 50,
                "interpretation": {
                    "low": "Model is confident and coherent",
                    "medium": "Moderate confidence",
                    "high": "Model is uncertain or incoherent"
                }
            }
        }
        
        self.logger.info("AI Tutor initialized with comprehensive knowledge base")

    def explain_performance(self, task: str, results: List[Dict], evaluation: Optional[pd.DataFrame] = None) -> str:
        """
        Provide comprehensive explanation of model performance
        
        Args:
            task: Type of task (summarization, translation, chat)
            results: List of model results
            evaluation: Evaluation metrics DataFrame
            
        Returns:
            Detailed performance explanation
        """
        try:
            if not results:
                return "No results available to analyze."
            
            explanation_parts = []
            
            # Header
            explanation_parts.append(f"🎓 **AI Tutor Analysis for {task.title()}**\n")
            
            # Basic performance overview
            successful_models = [r for r in results if r.get("success", True) and r.get("output")]
            failed_models = [r for r in results if not r.get("success", True) or not r.get("output")]
            
            explanation_parts.append(f"**📊 Performance Overview:**")
            explanation_parts.append(f"- Successful models: {len(successful_models)}/{len(results)}")
            explanation_parts.append(f"- Failed models: {len(failed_models)}")
            
            if failed_models:
                explanation_parts.append(f"\n**❌ Failed Models:**")
                for model in failed_models:
                    error = model.get("error", "Unknown error")
                    explanation_parts.append(f"- {model['model']}: {error}")
            
            # Model-specific analysis
            if successful_models:
                explanation_parts.append(f"\n**🤖 Model Analysis:**")
                
                for result in successful_models:
                    model_analysis = self._analyze_single_model(result, task)
                    explanation_parts.append(model_analysis)
            
            # Evaluation metrics explanation
            if evaluation is not None and not evaluation.empty:
                metrics_explanation = self._explain_metrics(evaluation, task)
                explanation_parts.append(f"\n{metrics_explanation}")
            
            # Performance comparison
            if len(successful_models) > 1:
                comparison = self._compare_models(successful_models, evaluation)
                explanation_parts.append(f"\n{comparison}")
            
            # Recommendations
            recommendations = self._generate_recommendations(task, results, evaluation)
            explanation_parts.append(f"\n{recommendations}")
            
            # Educational insights
            insights = self._provide_insights(task, results, evaluation)
            explanation_parts.append(f"\n{insights}")
            
            full_explanation = "\n".join(explanation_parts)
            
            self.logger.info(f"Generated comprehensive explanation for {task} task")
            return full_explanation
            
        except Exception as e:
            self.logger.error(f"Error generating performance explanation: {e}")
            return f"Error generating explanation: {str(e)}"

    def _analyze_single_model(self, result: Dict, task: str) -> str:
        """Analyze performance of a single model"""
        model_name = result.get("model", "Unknown")
        output = result.get("output", "")
        inference_time = result.get("inference_time", 0)
        
        analysis_parts = []
        analysis_parts.append(f"\n**{model_name}:**")
        
        # Output analysis
        if output:
            word_count = len(output.split())
            char_count = len(output)
            
            analysis_parts.append(f"- Output length: {word_count} words, {char_count} characters")
            
            # Speed analysis
            if inference_time > 0:
                words_per_second = word_count / inference_time
                analysis_parts.append(f"- Speed: {words_per_second:.1f} words/second ({inference_time:.2f}s total)")
                
                if words_per_second > 20:
                    speed_assessment = "Very fast"
                elif words_per_second > 10:
                    speed_assessment = "Fast"
                elif words_per_second > 5:
                    speed_assessment = "Moderate"
                else:
                    speed_assessment = "Slow"
                
                analysis_parts.append(f"- Speed assessment: {speed_assessment}")
            
            # Content analysis
            content_analysis = self._analyze_content_quality(output, task)
            analysis_parts.append(content_analysis)
        
        # Quality issues
        quality_issues = result.get("quality_issues", [])
        if quality_issues:
            analysis_parts.append(f"- Quality concerns: {', '.join(quality_issues)}")
        
        return "\n".join(analysis_parts)

    def _analyze_content_quality(self, output: str, task: str) -> str:
        """Analyze content quality"""
        analysis = []
        
        # Sentence structure
        sentences = [s.strip() for s in output.split('.') if s.strip()]
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        analysis.append(f"- Sentence structure: {len(sentences)} sentences, avg {avg_sentence_length:.1f} words/sentence")
        
        # Vocabulary diversity
        words = output.lower().split()
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words) if words else 0
        
        if diversity_ratio > 0.7:
            vocab_assessment = "High vocabulary diversity"
        elif diversity_ratio > 0.5:
            vocab_assessment = "Moderate vocabulary diversity"
        else:
            vocab_assessment = "Low vocabulary diversity"
        
        analysis.append(f"- Vocabulary: {vocab_assessment} ({diversity_ratio:.2f} ratio)")
        
        # Task-specific analysis
        if task == "summarization":
            # Check for summarization quality indicators
            if any(word in output.lower() for word in ["summary", "conclusion", "overview", "key points"]):
                analysis.append("- Contains summarization keywords (good)")
            
        elif task == "translation":
            # Check for translation quality indicators
            if output.count("...") > 2:
                analysis.append("- Multiple ellipses may indicate uncertainty")
        
        return "\n".join(analysis)

    def _explain_metrics(self, evaluation: pd.DataFrame, task: str) -> str:
        """Explain evaluation metrics in detail"""
        explanation_parts = []
        explanation_parts.append("**📏 Metrics Explanation:**")
        
        # Get metrics from dataframe columns
        metric_columns = [col for col in evaluation.columns if col not in ["model", "inference_time"]]
        
        for metric in metric_columns:
            if metric in self.metric_explanations:
                metric_info = self.metric_explanations[metric]
                explanation_parts.append(f"\n**{metric.upper()}:**")
                explanation_parts.append(f"- {metric_info['description']}")
                explanation_parts.append(f"- Range: {metric_info['range']}")
                
                # Show actual scores
                scores = evaluation[metric].values
                best_score = scores.max() if metric != "perplexity" else scores.min()
                worst_score = scores.min() if metric != "perplexity" else scores.max()
                
                explanation_parts.append(f"- Best score: {best_score:.3f}")
                explanation_parts.append(f"- Worst score: {worst_score:.3f}")
                
                # Interpretation
                if metric == "perplexity":
                    if best_score < metric_info["good_score"]:
                        interpretation = metric_info["interpretation"]["low"]
                    elif best_score < metric_info["good_score"] * 2:
                        interpretation = metric_info["interpretation"]["medium"]
                    else:
                        interpretation = metric_info["interpretation"]["high"]
                else:
                    if best_score > metric_info["good_score"]:
                        interpretation = metric_info["interpretation"]["high"]
                    elif best_score > metric_info["good_score"] * 0.5:
                        interpretation = metric_info["interpretation"]["medium"]
                    else:
                        interpretation = metric_info["interpretation"]["low"]
                
                explanation_parts.append(f"- Assessment: {interpretation}")
        
        return "\n".join(explanation_parts)

    def _compare_models(self, results: List[Dict], evaluation: Optional[pd.DataFrame]) -> str:
        """Compare model performances"""
        comparison_parts = []
        comparison_parts.append("**⚖️ Model Comparison:**")
        
        # Speed comparison
        inference_times = [(r["model"], r.get("inference_time", 0)) for r in results]
        fastest = min(inference_times, key=lambda x: x[1])
        slowest = max(inference_times, key=lambda x: x[1])
        
        comparison_parts.append(f"- Fastest: {fastest[0]} ({fastest[1]:.2f}s)")
        comparison_parts.append(f"- Slowest: {slowest[0]} ({slowest[1]:.2f}s)")
        
        # Output length comparison
        output_lengths = [(r["model"], len(r.get("output", "").split())) for r in results]
        longest = max(output_lengths, key=lambda x: x[1])
        shortest = min(output_lengths, key=lambda x: x[1])
        
        comparison_parts.append(f"- Most verbose: {longest[0]} ({longest[1]} words)")
        comparison_parts.append(f"- Most concise: {shortest[0]} ({shortest[1]} words)")
        
        # Metric-based comparison
        if evaluation is not None and not evaluation.empty:
            metric_columns = [col for col in evaluation.columns if col not in ["model", "inference_time"]]
            
            for metric in metric_columns:
                if metric == "perplexity":
                    best_idx = evaluation[metric].idxmin()
                else:
                    best_idx = evaluation[metric].idxmax()
                
                best_model = evaluation.loc[best_idx, "model"]
                best_score = evaluation.loc[best_idx, metric]
                
                comparison_parts.append(f"- Best {metric}: {best_model} ({best_score:.3f})")
        
        return "\n".join(comparison_parts)

    def _generate_recommendations(self, task: str, results: List[Dict], evaluation: Optional[pd.DataFrame]) -> str:
        """Generate improvement recommendations"""
        recommendations = []
        recommendations.append("**💡 Recommendations:**")
        
        successful_results = [r for r in results if r.get("success", True) and r.get("output")]
        
        if not successful_results:
            recommendations.append("- Fix model errors before proceeding with optimization")
            return "\n".join(recommendations)
        
        # Speed recommendations
        avg_time = np.mean([r.get("inference_time", 0) for r in successful_results])
        if avg_time > 10:
            recommendations.append("- Consider using faster models or optimizing inference for better speed")
        
        # Output quality recommendations
        if task == "summarization":
            avg_length = np.mean([len(r.get("output", "").split()) for r in successful_results])
            if avg_length > 100:
                recommendations.append("- Summaries are quite long; consider reducing max_tokens for more concise outputs")
            elif avg_length < 20:
                recommendations.append("- Summaries are very short; consider increasing min_tokens for more detailed outputs")
        
        elif task == "translation":
            # Check for quality issues
            quality_issues = []
            for result in successful_results:
                issues = result.get("quality_issues", [])
                quality_issues.extend(issues)
            
            if "Contains many English words" in quality_issues:
                recommendations.append("- Translations contain English words; try more specific prompts or different models")
            if "Translation much longer than original" in quality_issues:
                recommendations.append("- Translations are too verbose; consider adjusting temperature or using more direct prompts")
        
        # Metric-based recommendations
        if evaluation is not None and not evaluation.empty:
            metric_columns = [col for col in evaluation.columns if col not in ["model", "inference_time"]]
            
            for metric in metric_columns:
                if metric in self.metric_explanations:
                    scores = evaluation[metric].values
                    avg_score = np.mean(scores)
                    good_score = self.metric_explanations[metric]["good_score"]
                    
                    if metric == "perplexity":
                        if avg_score > good_score * 2:
                            recommendations.append(f"- High perplexity indicates model uncertainty; try lower temperature or different prompts")
                    else:
                        if avg_score < good_score * 0.5:
                            recommendations.append(f"- Low {metric} scores suggest poor quality; consider using reference texts or adjusting parameters")
        
        # Parameter recommendations
        recommendations.append("\n**🎛️ Parameter Tuning Tips:**")
        recommendations.append("- Lower temperature (0.3-0.7) for more focused, consistent outputs")
        recommendations.append("- Higher temperature (0.8-1.2) for more creative, diverse outputs")
        recommendations.append("- Adjust top_p (0.7-0.95) to control output diversity")
        recommendations.append("- Use min_tokens to ensure adequate output length")
        
        return "\n".join(recommendations)

    def _provide_insights(self, task: str, results: List[Dict], evaluation: Optional[pd.DataFrame]) -> str:
        """Provide educational insights about AI and the task"""
        insights = []
        insights.append("**🧠 Educational Insights:**")
        
        # Task-specific insights
        if task == "summarization":
            insights.append("\n**About Summarization:**")
            insights.append("- Extractive summarization selects important sentences from the original text")
            insights.append("- Abstractive summarization generates new sentences that capture key ideas")
            insights.append("- Modern AI models typically use abstractive approaches for more natural summaries")
            insights.append("- ROUGE metrics compare n-gram overlap, while BERT-based metrics capture semantic similarity")
        
        elif task == "translation":
            insights.append("\n**About Translation:**")
            insights.append("- Neural machine translation uses attention mechanisms to focus on relevant parts")
            insights.append("- Quality depends on training data diversity and language pair similarity")
            insights.append("- Context and domain-specific terminology significantly impact accuracy")
            insights.append("- BLEU scores measure precision, while METEOR considers recall and synonyms")
        
        elif task == "chat":
            insights.append("\n**About Conversational AI:**")
            insights.append("- Language models predict the next token based on context and training")
            insights.append("- Temperature controls randomness in token selection")
            insights.append("- Conversation history provides context for coherent responses")
            insights.append("- Perplexity measures how 'surprised' the model is by the text")
        
        # Model insights
        insights.append("\n**About AI Models:**")
        insights.append("- Larger models generally perform better but are slower and more expensive")
        insights.append("- Fine-tuning on specific domains can improve performance significantly")
        insights.append("- Model performance varies by task, language, and domain")
        insights.append("- Ensemble approaches (model fusion) can combine strengths of different models")
        
        # Performance insights
        if evaluation is not None and not evaluation.empty:
            insights.append("\n**Performance Patterns:**")
            
            # Speed vs quality tradeoff
            if "inference_time" in evaluation.columns:
                speed_quality_insight = self._analyze_speed_quality_tradeoff(evaluation)
                insights.append(speed_quality_insight)
        
        return "\n".join(insights)

    def _analyze_speed_quality_tradeoff(self, evaluation: pd.DataFrame) -> str:
        """Analyze the tradeoff between speed and quality"""
        try:
            # Get quality metrics (exclude speed and model name)
            quality_metrics = [col for col in evaluation.columns if col not in ["model", "inference_time"]]
            
            if not quality_metrics or "inference_time" not in evaluation.columns:
                return "- Speed-quality analysis requires both performance metrics and timing data"
            
            # Calculate average quality score
            quality_scores = []
            for _, row in evaluation.iterrows():
                metric_values = []
                for metric in quality_metrics:
                    value = row[metric]
                    # Normalize perplexity (lower is better) vs other metrics (higher is better)
                    if metric == "perplexity":
                        normalized_value = 1 / (1 + value)  # Convert to 0-1 range where higher is better
                    else:
                        normalized_value = value
                    metric_values.append(normalized_value)
                
                avg_quality = np.mean(metric_values) if metric_values else 0
                quality_scores.append(avg_quality)
            
            # Analyze correlation between speed and quality
            times = evaluation["inference_time"].values
            correlation = np.corrcoef(times, quality_scores)[0, 1] if len(times) > 1 else 0
            
            if abs(correlation) < 0.3:
                return "- No strong correlation between speed and quality observed"
            elif correlation > 0.3:
                return "- Slower models tend to produce higher quality outputs (expected for larger models)"
            else:
                return "- Faster models show better quality (unusual - may indicate optimization benefits)"
                
        except Exception as e:
            return f"- Speed-quality analysis failed: {str(e)}"

    def explain_metric(self, metric_name: str, score: float) -> str:
        """
        Explain a specific metric and score
        
        Args:
            metric_name: Name of the metric
            score: The score value
            
        Returns:
            Detailed explanation of the metric and score
        """
        try:
            metric_name = metric_name.lower()
            
            if metric_name not in self.metric_explanations:
                return f"Unknown metric: {metric_name}"
            
            metric_info = self.metric_explanations[metric_name]
            
            explanation = []
            explanation.append(f"**{metric_name.upper()} Score: {score:.3f}**")
            explanation.append(f"Description: {metric_info['description']}")
            explanation.append(f"Range: {metric_info['range']}")
            
            # Interpret the score
            good_score = metric_info["good_score"]
            
            if metric_name == "perplexity":
                if score < good_score:
                    interpretation = metric_info["interpretation"]["low"]
                    quality = "Excellent"
                elif score < good_score * 2:
                    interpretation = metric_info["interpretation"]["medium"]
                    quality = "Good"
                else:
                    interpretation = metric_info["interpretation"]["high"]
                    quality = "Poor"
            else:
                if score > good_score:
                    interpretation = metric_info["interpretation"]["high"]
                    quality = "Excellent"
                elif score > good_score * 0.5:
                    interpretation = metric_info["interpretation"]["medium"]
                    quality = "Good"
                else:
                    interpretation = metric_info["interpretation"]["low"]
                    quality = "Poor"
            
            explanation.append(f"Quality Assessment: {quality}")
            explanation.append(f"Interpretation: {interpretation}")
            
            return "\n".join(explanation)
            
        except Exception as e:
            self.logger.error(f"Error explaining metric {metric_name}: {e}")
            return f"Error explaining metric: {str(e)}"

    def suggest_improvements(self, task: str, current_scores: Dict[str, float], target_improvement: float = 0.1) -> List[str]:
        """
        Suggest specific improvements to achieve better scores
        
        Args:
            task: Type of task
            current_scores: Dictionary of current metric scores
            target_improvement: Target improvement amount
            
        Returns:
            List of improvement suggestions
        """
        try:
            suggestions = []
            
            for metric, score in current_scores.items():
                if metric.lower() in self.metric_explanations:
                    metric_suggestions = self._get_metric_specific_suggestions(metric.lower(), score, task)
                    suggestions.extend(metric_suggestions)
            
            # Add general suggestions
            general_suggestions = self._get_general_improvement_suggestions(task)
            suggestions.extend(general_suggestions)
            
            return list(set(suggestions))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {e}")
            return [f"Error generating suggestions: {str(e)}"]

    def _get_metric_specific_suggestions(self, metric: str, score: float, task: str) -> List[str]:
        """Get suggestions specific to a metric"""
        suggestions = []
        metric_info = self.metric_explanations[metric]
        good_score = metric_info["good_score"]
        
        if metric in ["rouge1", "rouge2", "rougeL"]:
            if score < good_score * 0.5:
                suggestions.append("Use reference texts or examples to guide the model")
                suggestions.append("Try more specific prompts that include desired output characteristics")
                suggestions.append("Consider fine-tuning on domain-specific data")
        
        elif metric == "bertscore":
            if score < good_score:
                suggestions.append("Focus on semantic similarity rather than exact word matching")
                suggestions.append("Use prompts that encourage paraphrasing key concepts")
        
        elif metric in ["bleu", "meteor"]:
            if score < good_score:
                suggestions.append("Provide more context about the target language and domain")
                suggestions.append("Use translation-specific prompts and examples")
                suggestions.append("Consider post-editing for common translation errors")
        
        elif metric == "perplexity":
            if score > good_score * 2:
                suggestions.append("Lower the temperature for more confident predictions")
                suggestions.append("Provide more context in the prompt")
                suggestions.append("Use more specific and clear instructions")
        
        return suggestions

    def _get_general_improvement_suggestions(self, task: str) -> List[str]:
        """Get general improvement suggestions for a task"""
        suggestions = []
        
        if task == "summarization":
            suggestions.extend([
                "Experiment with different summary lengths",
                "Try prompt engineering with specific instructions (e.g., 'Focus on key findings')",
                "Use examples of good summaries in your prompts",
                "Consider multi-step summarization for very long texts"
            ])
        
        elif task == "translation":
            suggestions.extend([
                "Provide cultural context when relevant",
                "Specify the target audience (formal/informal)",
                "Include domain-specific terminology in prompts",
                "Use back-translation to verify quality"
            ])
        
        elif task == "chat":
            suggestions.extend([
                "Maintain conversation context and history",
                "Use system prompts to define the AI's role and behavior",
                "Experiment with different conversation styles",
                "Provide examples of desired response format"
            ])
        
        # Universal suggestions
        suggestions.extend([
            "A/B test different parameter settings",
            "Use ensemble methods to combine multiple models",
            "Collect user feedback for continuous improvement",
            "Monitor performance across different input types"
        ])
        
        return suggestions

    def generate_learning_report(self, task: str, results: List[Dict], evaluation: Optional[pd.DataFrame] = None) -> str:
        """
        Generate a comprehensive learning report
        
        Args:
            task: Type of task
            results: Model results
            evaluation: Evaluation metrics
            
        Returns:
            Comprehensive learning report
        """
        try:
            report_parts = []
            
            # Title
            report_parts.append("# 📚 AI Performance Learning Report")
            report_parts.append(f"**Task:** {task.title()}")
            report_parts.append(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_parts.append("---")
            
            # Executive Summary
            report_parts.append("## 📋 Executive Summary")
            summary = self._generate_executive_summary(results, evaluation)
            report_parts.append(summary)
            
            # Detailed Analysis
            detailed_analysis = self.explain_performance(task, results, evaluation)
            report_parts.append("## 🔍 Detailed Analysis")
            report_parts.append(detailed_analysis)
            
            # Learning Outcomes
            report_parts.append("## 🎯 Key Learning Outcomes")
            learning_outcomes = self._generate_learning_outcomes(task, results, evaluation)
            report_parts.append(learning_outcomes)
            
            # Action Items
            report_parts.append("## ✅ Recommended Actions")
            action_items = self._generate_action_items(task, results, evaluation)
            report_parts.append(action_items)
            
            # Further Reading
            report_parts.append("## 📖 Further Reading")
            reading_suggestions = self._generate_reading_suggestions(task)
            report_parts.append(reading_suggestions)
            
            return "\n\n".join(report_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating learning report: {e}")
            return f"Error generating learning report: {str(e)}"

    def _generate_executive_summary(self, results: List[Dict], evaluation: Optional[pd.DataFrame]) -> str:
        """Generate executive summary"""
        successful = len([r for r in results if r.get("success", True)])
        total = len(results)
        
        summary = [
            f"**Models Tested:** {total}",
            f"**Success Rate:** {successful}/{total} ({100*successful/total:.1f}%)",
        ]
        
        if evaluation is not None and not evaluation.empty:
            # Find best performing model
            metric_columns = [col for col in evaluation.columns if col not in ["model", "inference_time"]]
            if metric_columns:
                # Use first metric for ranking (could be improved)
                first_metric = metric_columns[0]
                if first_metric == "perplexity":
                    best_idx = evaluation[first_metric].idxmin()
                else:
                    best_idx = evaluation[first_metric].idxmax()
                
                best_model = evaluation.loc[best_idx, "model"]
                summary.append(f"**Best Performing Model:** {best_model}")
        
        return "\n".join(summary)

    def _generate_learning_outcomes(self, task: str, results: List[Dict], evaluation: Optional[pd.DataFrame]) -> str:
        """Generate learning outcomes"""
        outcomes = [
            "Through this analysis, you should have learned:",
            f"- How different AI models perform on {task} tasks",
            "- The importance of evaluation metrics in assessing quality",
            "- How parameter tuning affects model outputs",
            "- The trade-offs between speed and quality in AI systems"
        ]
        
        if evaluation is not None and not evaluation.empty:
            outcomes.append("- How to interpret quantitative performance metrics")
            outcomes.append("- The relationship between different evaluation approaches")
        
        return "\n".join(outcomes)

    def _generate_action_items(self, task: str, results: List[Dict], evaluation: Optional[pd.DataFrame]) -> str:
        """Generate actionable next steps"""
        actions = [
            "**Immediate Actions:**",
            "1. Review the model comparison results",
            "2. Test the recommended parameter adjustments",
            "3. Experiment with different prompt styles",
            "",
            "**Medium-term Actions:**",
            "1. Collect more diverse test cases",
            "2. Set up automated evaluation pipelines", 
            "3. Consider fine-tuning for your specific use case",
            "",
            "**Long-term Actions:**",
            "1. Monitor performance over time",
            "2. Build domain-specific evaluation datasets",
            "3. Explore advanced techniques like RAG or model ensembles"
        ]
        
        return "\n".join(actions)

    def _generate_reading_suggestions(self, task: str) -> str:
        """Generate reading suggestions for further learning"""
        suggestions = [
            "**Recommended Resources:**",
            "",
            "**General AI/NLP:**",
            "- 'Attention Is All You Need' (Transformer paper)",
            "- 'BERT: Pre-training of Deep Bidirectional Transformers'",
            "- OpenAI GPT series papers",
            "",
        ]
        
        if task == "summarization":
            suggestions.extend([
                "**Summarization-Specific:**",
                "- 'ROUGE: A Package for Automatic Evaluation of Summaries'",
                "- 'BERTScore: Evaluating Text Generation with BERT'",
                "- Recent papers on abstractive summarization",
            ])
        elif task == "translation":
            suggestions.extend([
                "**Translation-Specific:**",
                "- 'BLEU: a Method for Automatic Evaluation of Machine Translation'",
                "- 'METEOR: An Automatic Metric for MT Evaluation'",
                "- Neural Machine Translation research papers",
            ])
        elif task == "chat":
            suggestions.extend([
                "**Conversational AI:**",
                "- 'DialogPT: Large-Scale Generative Pre-training for Conversational Response'",
                "- 'LaMDA: Language Models for Dialog Applications'",
                "- Research on dialogue evaluation metrics",
            ])
        
        return "\n".join(suggestions)