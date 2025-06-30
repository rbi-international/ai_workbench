import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from utils.logger import setup_logger

class CostTracker:
    """
    Tracks API usage and costs for different models
    """
    
    # OpenAI pricing (per 1K tokens) - as of 2024
    OPENAI_PRICING = {
        "gpt-4o": {
            "input": 0.005,   # $0.005 per 1K input tokens
            "output": 0.015   # $0.015 per 1K output tokens
        },
        "gpt-4": {
            "input": 0.03,
            "output": 0.06
        },
        "gpt-3.5-turbo": {
            "input": 0.001,
            "output": 0.002
        }
    }
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(f"{__name__}_cost_tracker")
        
        # Create logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Usage file path
        self.usage_file = self.log_dir / "usage_data.json"
        
        # Load existing usage data
        self.usage = self._load_usage_data()
        
        # Load configuration
        try:
            import yaml
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.max_words = config.get("models", {}).get("openai", {}).get("max_output_words", 100)
        except (FileNotFoundError, yaml.YAMLError):
            self.max_words = 100
            
        self.logger.info("CostTracker initialized")

    def _load_usage_data(self) -> Dict[str, Any]:
        """Load usage data from file"""
        try:
            if self.usage_file.exists():
                with open(self.usage_file, 'r') as f:
                    data = json.load(f)
                self.logger.info("Loaded existing usage data")
                return data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.warning(f"Could not load usage data: {e}")
        
        # Return default structure
        return {
            "total_cost": 0.0,
            "models": {},
            "daily_usage": {},
            "last_updated": None
        }

    def _save_usage_data(self):
        """Save usage data to file"""
        try:
            self.usage["last_updated"] = datetime.now().isoformat()
            with open(self.usage_file, 'w') as f:
                json.dump(self.usage, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save usage data: {e}")

    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int, output_words: int):
        """
        Log usage for a model
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens  
            output_words: Number of words in output
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Initialize model data if not exists
            if model not in self.usage["models"]:
                self.usage["models"][model] = {
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_output_words": 0,
                    "total_requests": 0,
                    "total_cost": 0.0
                }
            
            # Initialize daily data if not exists
            if today not in self.usage["daily_usage"]:
                self.usage["daily_usage"][today] = {}
            if model not in self.usage["daily_usage"][today]:
                self.usage["daily_usage"][today][model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "output_words": 0,
                    "requests": 0,
                    "cost": 0.0
                }
            
            # Update totals
            self.usage["models"][model]["total_prompt_tokens"] += prompt_tokens
            self.usage["models"][model]["total_completion_tokens"] += completion_tokens
            self.usage["models"][model]["total_output_words"] += output_words
            self.usage["models"][model]["total_requests"] += 1
            
            # Update daily totals
            daily_model = self.usage["daily_usage"][today][model]
            daily_model["prompt_tokens"] += prompt_tokens
            daily_model["completion_tokens"] += completion_tokens
            daily_model["output_words"] += output_words
            daily_model["requests"] += 1
            
            # Calculate cost for OpenAI models
            cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
            if cost > 0:
                self.usage["models"][model]["total_cost"] += cost
                daily_model["cost"] += cost
                self.usage["total_cost"] += cost
            
            # Check word limit
            if output_words > self.max_words:
                self.logger.warning(
                    f"Output for {model} exceeds word limit: {output_words} > {self.max_words} words"
                )
            
            # Log usage info
            self.logger.info(
                f"Usage logged - Model: {model}, "
                f"Tokens: {prompt_tokens}+{completion_tokens}, "
                f"Words: {output_words}, Cost: ${cost:.4f}"
            )
            
            # Save data
            self._save_usage_data()
            
        except Exception as e:
            self.logger.error(f"Error logging usage: {e}")

    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for API usage
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Cost in USD
        """
        # Find pricing for the model
        pricing = None
        for price_model, price_data in self.OPENAI_PRICING.items():
            if price_model in model.lower():
                pricing = price_data
                break
        
        if not pricing:
            return 0.0  # No pricing data for this model
        
        # Calculate cost (pricing is per 1K tokens)
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost

    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary"""
        try:
            summary = {
                "total_cost": self.usage["total_cost"],
                "total_requests": sum(
                    model_data["total_requests"] 
                    for model_data in self.usage["models"].values()
                ),
                "models": {}
            }
            
            for model, data in self.usage["models"].items():
                summary["models"][model] = {
                    "requests": data["total_requests"],
                    "total_tokens": data["total_prompt_tokens"] + data["total_completion_tokens"],
                    "total_words": data["total_output_words"],
                    "cost": data["total_cost"]
                }
            
            return summary
        except Exception as e:
            self.logger.error(f"Error getting usage summary: {e}")
            return {"error": str(e)}

    def get_daily_usage(self, days: int = 7) -> Dict[str, Any]:
        """Get daily usage for the last N days"""
        try:
            from datetime import datetime, timedelta
            
            daily_data = {}
            today = datetime.now()
            
            for i in range(days):
                date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                if date in self.usage["daily_usage"]:
                    daily_data[date] = self.usage["daily_usage"][date]
                else:
                    daily_data[date] = {}
            
            return daily_data
        except Exception as e:
            self.logger.error(f"Error getting daily usage: {e}")
            return {"error": str(e)}

    def reset_usage(self):
        """Reset all usage data"""
        try:
            self.usage = {
                "total_cost": 0.0,
                "models": {},
                "daily_usage": {},
                "last_updated": None
            }
            self._save_usage_data()
            self.logger.info("Usage data reset")
        except Exception as e:
            self.logger.error(f"Error resetting usage: {e}")

    def check_budget(self, daily_limit: float = 10.0, monthly_limit: float = 100.0) -> Dict[str, Any]:
        """
        Check if usage is within budget limits
        
        Args:
            daily_limit: Daily spending limit in USD
            monthly_limit: Monthly spending limit in USD
            
        Returns:
            Budget status
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            this_month = datetime.now().strftime("%Y-%m")
            
            # Calculate daily spending
            daily_cost = 0.0
            if today in self.usage["daily_usage"]:
                for model_data in self.usage["daily_usage"][today].values():
                    daily_cost += model_data.get("cost", 0.0)
            
            # Calculate monthly spending
            monthly_cost = 0.0
            for date, daily_data in self.usage["daily_usage"].items():
                if date.startswith(this_month):
                    for model_data in daily_data.values():
                        monthly_cost += model_data.get("cost", 0.0)
            
            return {
                "daily": {
                    "spent": daily_cost,
                    "limit": daily_limit,
                    "remaining": daily_limit - daily_cost,
                    "over_budget": daily_cost > daily_limit
                },
                "monthly": {
                    "spent": monthly_cost,
                    "limit": monthly_limit,
                    "remaining": monthly_limit - monthly_cost,
                    "over_budget": monthly_cost > monthly_limit
                }
            }
        except Exception as e:
            self.logger.error(f"Error checking budget: {e}")
            return {"error": str(e)}