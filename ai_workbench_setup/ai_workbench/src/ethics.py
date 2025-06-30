from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from utils.logger import setup_logger
import yaml
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional, Tuple
import re
import json
from datetime import datetime

load_dotenv()

class EthicsAnalyzer:
    """
    Enhanced ethics analyzer for responsible AI usage with comprehensive
    sentiment, toxicity, bias, and safety analysis
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            ethics_config = config.get("ethics", {})
            self.sentiment_threshold = ethics_config.get("sentiment_threshold", 0.5)
            self.toxicity_threshold = ethics_config.get("toxicity_threshold", 0.7)
            self.enabled = ethics_config.get("enabled", True)
            
        except Exception as e:
            self.logger.warning(f"Could not load ethics configuration: {e}")
            self.sentiment_threshold = 0.5
            self.toxicity_threshold = 0.7
            self.enabled = True
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize toxicity analyzer with fallback
        self.toxicity_analyzer = None
        self._init_toxicity_analyzer()
        
        # Define harmful content patterns
        self.harmful_patterns = self._load_harmful_patterns()
        
        # Define bias detection patterns
        self.bias_patterns = self._load_bias_patterns()
        
        # Safety keywords
        self.safety_keywords = self._load_safety_keywords()
        
        self.logger.info("Ethics analyzer initialized with comprehensive safety checks")

    def _init_toxicity_analyzer(self):
        """Initialize toxicity analyzer with fallback options"""
        try:
            # Try to import and initialize Detoxify
            from detoxify import Detoxify
            self.toxicity_analyzer = Detoxify(model_type='original', device='cpu')
            self.logger.info("Detoxify toxicity analyzer initialized")
            
        except ImportError:
            self.logger.warning("Detoxify not available, using pattern-based toxicity detection")
            self.toxicity_analyzer = None
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Detoxify: {e}, using pattern-based detection")
            self.toxicity_analyzer = None

    def _load_harmful_patterns(self) -> List[str]:
        """Load patterns for harmful content detection"""
        return [
            # Violence and threats
            r'\b(kill|murder|assassinate|eliminate|destroy|bomb|attack|violence)\b',
            r'\b(threat|threaten|harm|hurt|damage|injury|wound)\b',
            r'\b(weapon|gun|knife|explosive|ammunition)\b',
            
            # Hate speech indicators
            r'\b(hate|despise|loathe|racist|sexist|bigot)\b',
            r'\b(inferior|superior|subhuman|worthless)\b',
            
            # Self-harm indicators
            r'\b(suicide|self[\-\s]harm|cut[\s]myself|end[\s]it[\s]all)\b',
            r'\b(overdose|jump[\s]off|hang[\s]myself)\b',
            
            # Illegal activities
            r'\b(fraud|scam|steal|rob|hack|piracy|counterfeit)\b',
            r'\b(drugs|cocaine|heroin|marijuana|illegal[\s]substance)\b',
            
            # Sexual content (basic detection)
            r'\b(sexual|explicit|adult|pornographic|erotic)\b',
            
            # Discrimination
            r'\b(discriminate|prejudice|stereotype|bias)\b'
        ]

    def _load_bias_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for bias detection"""
        return {
            "gender": [
                r'\b(men|women|male|female|boy|girl|guy|lady)\s+(are|should|must|can\'t|cannot)\b',
                r'\b(he|she)\s+(always|never|typically|usually)\b',
                r'\blike\s+a\s+(man|woman|girl|boy)\b'
            ],
            "racial": [
                r'\b(white|black|asian|hispanic|latino|african)\s+people\s+(are|should|must)\b',
                r'\b(race|ethnic|minority|majority)\s+(superior|inferior|better|worse)\b'
            ],
            "religious": [
                r'\b(christian|muslim|jewish|hindu|buddhist|atheist)\s+(are|should|must)\b',
                r'\b(religion|faith|belief)\s+(wrong|evil|bad|good)\b'
            ],
            "age": [
                r'\b(young|old|elderly|teenager|millennial|boomer)\s+people\s+(are|should|must)\b',
                r'\b(age|generation)\s+(problem|issue|fault)\b'
            ],
            "socioeconomic": [
                r'\b(poor|rich|wealthy|homeless|privileged)\s+people\s+(are|should|must)\b',
                r'\b(class|income|poverty)\s+(defines|determines|makes)\b'
            ]
        }

    def _load_safety_keywords(self) -> Dict[str, List[str]]:
        """Load safety-related keywords for monitoring"""
        return {
            "privacy_concern": ["personal information", "private data", "ssn", "social security", "credit card", "password", "login"],
            "misinformation": ["fake news", "conspiracy", "hoax", "false information", "misleading", "propaganda"],
            "manipulation": ["manipulate", "deceive", "trick", "fool", "mislead", "exploit", "advantage"],
            "extremism": ["radical", "extremist", "terrorist", "militia", "supremacist", "revolution"],
            "financial_harm": ["investment scam", "ponzi scheme", "get rich quick", "guaranteed profit", "financial advice"]
        }

    def analyze(self, outputs: List[str], context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Comprehensive ethical analysis of AI outputs
        
        Args:
            outputs: List of AI-generated outputs to analyze
            context: Optional context about the use case
            
        Returns:
            List of analysis results for each output
        """
        if not self.enabled:
            return [{"enabled": False} for _ in outputs]
        
        try:
            results = []
            
            for i, output in enumerate(outputs):
                if not output or not isinstance(output, str):
                    results.append({
                        "output_index": i,
                        "error": "Empty or invalid output",
                        "safe": True
                    })
                    continue
                
                # Comprehensive analysis
                analysis_result = self._analyze_single_output(output, context)
                analysis_result["output_index"] = i
                
                results.append(analysis_result)
            
            # Generate summary
            summary = self._generate_analysis_summary(results)
            
            # Log analysis
            self._log_analysis(results, summary)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in ethics analysis: {e}")
            return [{"error": str(e), "safe": False} for _ in outputs]

    def _analyze_single_output(self, output: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze a single output for ethical concerns"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "output_length": len(output),
            "word_count": len(output.split()),
            "safe": True,
            "risk_level": "low",
            "warnings": [],
            "recommendations": []
        }
        
        # Sentiment analysis
        sentiment_result = self._analyze_sentiment(output)
        result["sentiment"] = sentiment_result
        
        # Toxicity analysis
        toxicity_result = self._analyze_toxicity(output)
        result["toxicity"] = toxicity_result
        
        # Harmful content detection
        harmful_result = self._detect_harmful_content(output)
        result["harmful_content"] = harmful_result
        
        # Bias detection
        bias_result = self._detect_bias(output)
        result["bias"] = bias_result
        
        # Safety keyword analysis
        safety_result = self._analyze_safety_keywords(output)
        result["safety_keywords"] = safety_result
        
        # Privacy concern detection
        privacy_result = self._detect_privacy_concerns(output)
        result["privacy"] = privacy_result
        
        # Overall risk assessment
        risk_assessment = self._assess_overall_risk(result)
        result.update(risk_assessment)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result)
        result["recommendations"] = recommendations
        
        return result

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER"""
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Determine sentiment category
            compound = scores["compound"]
            if compound >= 0.05:
                sentiment_category = "positive"
            elif compound <= -0.05:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"
            
            # Check for extreme sentiment
            extreme_threshold = self.sentiment_threshold
            is_extreme = abs(compound) > extreme_threshold
            
            result = {
                "scores": scores,
                "category": sentiment_category,
                "is_extreme": is_extreme,
                "compound_score": compound
            }
            
            # Add warnings for extreme sentiment
            if is_extreme:
                if compound > extreme_threshold:
                    result["warning"] = f"Very positive sentiment detected (compound: {compound:.3f})"
                else:
                    result["warning"] = f"Very negative sentiment detected (compound: {compound:.3f})"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_toxicity(self, text: str) -> Dict[str, Any]:
        """Analyze toxicity using Detoxify or pattern-based approach"""
        try:
            if self.toxicity_analyzer:
                # Use Detoxify
                predictions = self.toxicity_analyzer.predict(text)
                
                # Check thresholds
                warnings = []
                high_toxicity = False
                
                for category, score in predictions.items():
                    if score > self.toxicity_threshold:
                        warnings.append(f"High {category}: {score:.3f}")
                        high_toxicity = True
                
                return {
                    "scores": {k: float(v) for k, v in predictions.items()},
                    "high_toxicity": high_toxicity,
                    "warnings": warnings,
                    "method": "detoxify"
                }
            else:
                # Pattern-based toxicity detection
                return self._pattern_based_toxicity(text)
                
        except Exception as e:
            self.logger.error(f"Toxicity analysis failed: {e}")
            return {"error": str(e), "method": "failed"}

    def _pattern_based_toxicity(self, text: str) -> Dict[str, Any]:
        """Pattern-based toxicity detection as fallback"""
        toxic_patterns = [
            r'\b(stupid|idiot|moron|dumb|retard)\b',
            r'\b(hate|kill|die|death)\b',
            r'\b(f\*ck|sh\*t|damn|hell)\b',
            r'\b(racist|sexist|bigot)\b'
        ]
        
        text_lower = text.lower()
        matches = []
        
        for pattern in toxic_patterns:
            if re.search(pattern, text_lower):
                matches.append(pattern)
        
        toxicity_score = min(1.0, len(matches) / 5.0)  # Normalize
        
        return {
            "scores": {"toxicity": toxicity_score},
            "high_toxicity": toxicity_score > self.toxicity_threshold,
            "warnings": [f"Potentially toxic content detected"] if matches else [],
            "method": "pattern_based",
            "matched_patterns": len(matches)
        }

    def _detect_harmful_content(self, text: str) -> Dict[str, Any]:
        """Detect harmful content using pattern matching"""
        try:
            text_lower = text.lower()
            detected_categories = []
            specific_matches = []
            
            for pattern in self.harmful_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    detected_categories.append(pattern)
                    specific_matches.extend(matches)
            
            has_harmful_content = len(detected_categories) > 0
            risk_score = min(1.0, len(detected_categories) / 10.0)
            
            return {
                "detected": has_harmful_content,
                "risk_score": risk_score,
                "categories_matched": len(detected_categories),
                "specific_matches": list(set(specific_matches)),
                "warning": "Potentially harmful content detected" if has_harmful_content else None
            }
            
        except Exception as e:
            self.logger.error(f"Harmful content detection failed: {e}")
            return {"error": str(e)}

    def _detect_bias(self, text: str) -> Dict[str, Any]:
        """Detect potential bias in text"""
        try:
            detected_biases = {}
            overall_bias_score = 0.0
            
            for bias_type, patterns in self.bias_patterns.items():
                matches = []
                for pattern in patterns:
                    found = re.findall(pattern, text, re.IGNORECASE)
                    matches.extend(found)
                
                if matches:
                    bias_score = min(1.0, len(matches) / 3.0)
                    detected_biases[bias_type] = {
                        "score": bias_score,
                        "matches": matches[:5],  # Limit displayed matches
                        "warning": f"Potential {bias_type} bias detected"
                    }
                    overall_bias_score += bias_score
            
            overall_bias_score = min(1.0, overall_bias_score / len(self.bias_patterns))
            
            return {
                "overall_score": overall_bias_score,
                "detected_biases": detected_biases,
                "has_bias": len(detected_biases) > 0,
                "bias_types": list(detected_biases.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Bias detection failed: {e}")
            return {"error": str(e)}

    def _analyze_safety_keywords(self, text: str) -> Dict[str, Any]:
        """Analyze safety-related keywords"""
        try:
            text_lower = text.lower()
            detected_concerns = {}
            
            for concern_type, keywords in self.safety_keywords.items():
                matches = []
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        matches.append(keyword)
                
                if matches:
                    detected_concerns[concern_type] = {
                        "keywords_found": matches,
                        "severity": len(matches) / len(keywords),
                        "warning": f"Safety concern: {concern_type}"
                    }
            
            overall_safety_score = 1.0 - min(1.0, len(detected_concerns) / len(self.safety_keywords))
            
            return {
                "safety_score": overall_safety_score,
                "concerns": detected_concerns,
                "has_safety_issues": len(detected_concerns) > 0,
                "concern_types": list(detected_concerns.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Safety keyword analysis failed: {e}")
            return {"error": str(e)}

    def _detect_privacy_concerns(self, text: str) -> Dict[str, Any]:
        """Detect potential privacy violations"""
        try:
            privacy_patterns = {
                "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "phone": r'\b(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
                "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                "address": r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b'
            }
            
            detected_pii = {}
            
            for pii_type, pattern in privacy_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    detected_pii[pii_type] = {
                        "count": len(matches),
                        "examples": matches[:2],  # Show first 2 examples
                        "warning": f"Potential {pii_type} detected"
                    }
            
            privacy_risk = min(1.0, len(detected_pii) / 3.0)
            
            return {
                "privacy_risk": privacy_risk,
                "detected_pii": detected_pii,
                "has_privacy_concerns": len(detected_pii) > 0,
                "pii_types": list(detected_pii.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Privacy concern detection failed: {e}")
            return {"error": str(e)}

    def _assess_overall_risk(self, analysis: Dict) -> Dict[str, Any]:
        """Assess overall risk level based on all analyses"""
        try:
            risk_factors = []
            risk_score = 0.0
            
            # Sentiment risk
            sentiment = analysis.get("sentiment", {})
            if sentiment.get("is_extreme"):
                risk_factors.append("extreme_sentiment")
                risk_score += 0.2
            
            # Toxicity risk
            toxicity = analysis.get("toxicity", {})
            if toxicity.get("high_toxicity"):
                risk_factors.append("high_toxicity")
                risk_score += 0.4
            
            # Harmful content risk
            harmful = analysis.get("harmful_content", {})
            if harmful.get("detected"):
                risk_factors.append("harmful_content")
                risk_score += 0.3
            
            # Bias risk
            bias = analysis.get("bias", {})
            if bias.get("has_bias"):
                risk_factors.append("bias_detected")
                risk_score += 0.2
            
            # Safety concerns
            safety = analysis.get("safety_keywords", {})
            if safety.get("has_safety_issues"):
                risk_factors.append("safety_concerns")
                risk_score += 0.3
            
            # Privacy concerns
            privacy = analysis.get("privacy", {})
            if privacy.get("has_privacy_concerns"):
                risk_factors.append("privacy_violations")
                risk_score += 0.4
            
            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "high"
                safe = False
            elif risk_score >= 0.4:
                risk_level = "medium"
                safe = False
            elif risk_score >= 0.1:
                risk_level = "low"
                safe = True
            else:
                risk_level = "minimal"
                safe = True
            
            return {
                "risk_score": min(1.0, risk_score),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "safe": safe,
                "requires_review": risk_score >= 0.4
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return {"error": str(e), "safe": False}

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Risk level recommendations
        risk_level = analysis.get("risk_level", "minimal")
        
        if risk_level == "high":
            recommendations.append("🚨 HIGH RISK: Consider blocking or significantly modifying this output")
            recommendations.append("📋 Manual review required before use")
        elif risk_level == "medium":
            recommendations.append("⚠️ MEDIUM RISK: Review and consider modifications")
            recommendations.append("🔍 Additional safety checks recommended")
        elif risk_level == "low":
            recommendations.append("ℹ️ LOW RISK: Minor concerns detected, monitor usage")
        
        # Specific recommendations
        if analysis.get("sentiment", {}).get("is_extreme"):
            recommendations.append("🎭 Consider balancing extreme sentiment in the output")
        
        if analysis.get("toxicity", {}).get("high_toxicity"):
            recommendations.append("🧹 Remove or rephrase toxic language")
        
        if analysis.get("harmful_content", {}).get("detected"):
            recommendations.append("🛡️ Remove harmful content references")
        
        if analysis.get("bias", {}).get("has_bias"):
            bias_types = analysis["bias"]["bias_types"]
            recommendations.append(f"⚖️ Address potential bias: {', '.join(bias_types)}")
        
        if analysis.get("privacy", {}).get("has_privacy_concerns"):
            recommendations.append("🔒 Remove or anonymize personal information")
        
        if analysis.get("safety_keywords", {}).get("has_safety_issues"):
            recommendations.append("🛟 Review safety-related content for appropriateness")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ Content appears ethically appropriate")
        
        recommendations.append("📚 Always consider context and intended audience")
        recommendations.append("🔄 Regular monitoring and updates to safety measures recommended")
        
        return recommendations

    def _generate_analysis_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary of all analyses"""
        try:
            total_outputs = len(results)
            safe_outputs = sum(1 for r in results if r.get("safe", True))
            high_risk_outputs = sum(1 for r in results if r.get("risk_level") == "high")
            
            # Aggregate risk factors
            all_risk_factors = []
            for result in results:
                risk_factors = result.get("risk_factors", [])
                all_risk_factors.extend(risk_factors)
            
            # Count common issues
            risk_factor_counts = {}
            for factor in all_risk_factors:
                risk_factor_counts[factor] = risk_factor_counts.get(factor, 0) + 1
            
            return {
                "total_outputs": total_outputs,
                "safe_outputs": safe_outputs,
                "unsafe_outputs": total_outputs - safe_outputs,
                "high_risk_outputs": high_risk_outputs,
                "safety_rate": safe_outputs / total_outputs if total_outputs > 0 else 1.0,
                "common_risk_factors": risk_factor_counts,
                "overall_assessment": "SAFE" if safe_outputs == total_outputs else "REQUIRES_ATTENTION"
            }
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return {"error": str(e)}

    def _log_analysis(self, results: List[Dict], summary: Dict):
        """Log analysis results for monitoring"""
        try:
            # Log summary
            self.logger.info(f"Ethics analysis complete: {summary}")
            
            # Log high-risk outputs
            for i, result in enumerate(results):
                if result.get("risk_level") == "high":
                    self.logger.warning(f"High-risk output detected (index {i}): {result.get('risk_factors', [])}")
                elif result.get("risk_level") == "medium":
                    self.logger.info(f"Medium-risk output detected (index {i}): {result.get('risk_factors', [])}")
            
        except Exception as e:
            self.logger.error(f"Error logging analysis: {e}")

    def is_content_safe(self, text: str) -> Tuple[bool, str]:
        """
        Quick safety check for a single text
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (is_safe, reason)
        """
        try:
            if not text or not isinstance(text, str):
                return True, "Empty or invalid text"
            
            # Quick checks
            result = self._analyze_single_output(text)
            
            is_safe = result.get("safe", True)
            risk_level = result.get("risk_level", "minimal")
            risk_factors = result.get("risk_factors", [])
            
            if not is_safe:
                reason = f"Risk level: {risk_level}, factors: {', '.join(risk_factors)}"
            else:
                reason = "Content appears safe"
            
            return is_safe, reason
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False, f"Safety check error: {str(e)}"

    def get_ethics_report(self, results: List[Dict]) -> str:
        """
        Generate a comprehensive ethics report
        
        Args:
            results: Analysis results
            
        Returns:
            Formatted ethics report
        """
        try:
            if not results:
                return "No analysis results available"
            
            summary = self._generate_analysis_summary(results)
            
            report_parts = []
            
            # Header
            report_parts.append("# 🛡️ AI Ethics Analysis Report")
            report_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_parts.append(f"**Total Outputs Analyzed:** {summary['total_outputs']}")
            report_parts.append("---")
            
            # Executive Summary
            report_parts.append("## 📋 Executive Summary")
            safety_rate = summary["safety_rate"] * 100
            report_parts.append(f"**Safety Rate:** {safety_rate:.1f}% ({summary['safe_outputs']}/{summary['total_outputs']})")
            report_parts.append(f"**Overall Assessment:** {summary['overall_assessment']}")
            
            if summary["high_risk_outputs"] > 0:
                report_parts.append(f"**⚠️ High Risk Outputs:** {summary['high_risk_outputs']}")
            
            # Common Issues
            if summary["common_risk_factors"]:
                report_parts.append("\n## 🔍 Common Risk Factors")
                for factor, count in sorted(summary["common_risk_factors"].items(), key=lambda x: x[1], reverse=True):
                    report_parts.append(f"- **{factor.replace('_', ' ').title()}:** {count} occurrence(s)")
            
            # Individual Analysis
            report_parts.append("\n## 📊 Individual Output Analysis")
            
            for i, result in enumerate(results):
                risk_level = result.get("risk_level", "minimal")
                risk_emoji = {"high": "🚨", "medium": "⚠️", "low": "ℹ️", "minimal": "✅"}
                
                report_parts.append(f"\n### Output {i+1} {risk_emoji.get(risk_level, '❓')}")
                report_parts.append(f"**Risk Level:** {risk_level.title()}")
                
                if result.get("risk_factors"):
                    report_parts.append(f"**Risk Factors:** {', '.join(result['risk_factors'])}")
                
                # Show specific warnings
                warnings = []
                for category in ["sentiment", "toxicity", "harmful_content", "bias", "privacy"]:
                    category_data = result.get(category, {})
                    if isinstance(category_data, dict) and category_data.get("warning"):
                        warnings.append(category_data["warning"])
                
                if warnings:
                    report_parts.append("**Warnings:**")
                    for warning in warnings:
                        report_parts.append(f"- {warning}")
                
                # Show recommendations
                recommendations = result.get("recommendations", [])
                if recommendations:
                    report_parts.append("**Recommendations:**")
                    for rec in recommendations[:3]:  # Show top 3
                        report_parts.append(f"- {rec}")
            
            # Overall Recommendations
            report_parts.append("\n## 💡 Overall Recommendations")
            
            if summary["safety_rate"] < 0.8:
                report_parts.append("- 🚨 **Critical:** Review AI model training and safety measures")
                report_parts.append("- 📋 Implement additional content filtering")
            elif summary["safety_rate"] < 0.95:
                report_parts.append("- ⚠️ **Important:** Enhance safety monitoring")
                report_parts.append("- 🔍 Consider additional bias detection measures")
            else:
                report_parts.append("- ✅ **Good:** Current safety measures appear effective")
                report_parts.append("- 🔄 Continue regular monitoring and updates")
            
            report_parts.append("- 📚 Regular ethics training for AI development team")
            report_parts.append("- 🛡️ Implement user feedback mechanisms for safety concerns")
            report_parts.append("- 📊 Regular auditing of AI outputs across different demographics")
            
            return "\n".join(report_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating ethics report: {e}")
            return f"Error generating ethics report: {str(e)}"

    def update_safety_thresholds(self, sentiment_threshold: float = None, toxicity_threshold: float = None):
        """Update safety thresholds"""
        if sentiment_threshold is not None:
            self.sentiment_threshold = max(0.0, min(1.0, sentiment_threshold))
            self.logger.info(f"Updated sentiment threshold to {self.sentiment_threshold}")
        
        if toxicity_threshold is not None:
            self.toxicity_threshold = max(0.0, min(1.0, toxicity_threshold))
            self.logger.info(f"Updated toxicity threshold to {self.toxicity_threshold}")

    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get current safety configuration and statistics"""
        return {
            "enabled": self.enabled,
            "sentiment_threshold": self.sentiment_threshold,
            "toxicity_threshold": self.toxicity_threshold,
            "toxicity_analyzer_available": self.toxicity_analyzer is not None,
            "harmful_patterns_count": len(self.harmful_patterns),
            "bias_categories": list(self.bias_patterns.keys()),
            "safety_keyword_categories": list(self.safety_keywords.keys())
        }