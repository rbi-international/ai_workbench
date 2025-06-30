from src.models.base_model import BaseModel
from utils.logger import setup_logger
from utils.helpers import validate_model_params
from typing import List, Dict, Any, Optional
import time
import json

class Chatter:
    """
    Chatter class that handles conversational chat across multiple models
    """
    
    def __init__(self, models: List[BaseModel]):
        self.logger = setup_logger(__name__)
        
        # Filter only available models
        self.models = [model for model in models if model.is_available()]
        
        if not self.models:
            self.logger.warning("No available models for chat")
        else:
            model_names = [model.get_name() for model in self.models]
            self.logger.info(f"Chatter initialized with models: {model_names}")
        
        # Conversation history storage
        self.conversation_history = {}

    def chat(self, messages: List[Dict], params: Dict[str, Any]) -> List[Dict]:
        """
        Generate chat responses using all available models
        
        Args:
            messages: List of conversation messages
            params: Generation parameters
            
        Returns:
            List of result dictionaries with model outputs
        """
        try:
            # Validate inputs
            if not messages or not messages[-1].get("content"):
                raise ValueError("Chat input cannot be empty")
            
            # Clean and validate messages
            messages = self._clean_messages(messages)
            params = validate_model_params(params)
            
            self.logger.info(f"Starting chat for {len(self.models)} models")
            self.logger.debug(f"Message history length: {len(messages)}")
            self.logger.debug(f"Last message: {messages[-1]['content'][:100]}...")
            self.logger.debug(f"Parameters: {params}")
            
            results = []
            
            for model in self.models:
                model_result = self._chat_with_model(model, messages, params)
                results.append(model_result)
            
            # Log summary statistics
            successful_results = [r for r in results if r.get("output") is not None]
            self.logger.info(
                f"Chat completed: {len(successful_results)}/{len(self.models)} models succeeded"
            )
            
            # Store conversation in history
            self._update_conversation_history(messages, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in chat: {e}")
            raise

    def _chat_with_model(self, model: BaseModel, messages: List[Dict], params: Dict[str, Any]) -> Dict:
        """
        Generate chat response with a single model
        
        Args:
            model: Model instance
            messages: List of conversation messages
            params: Generation parameters
            
        Returns:
            Result dictionary
        """
        model_name = model.get_name()
        start_time = time.time()
        
        try:
            self.logger.debug(f"Starting chat with {model_name}")
            
            # Check if model is available
            if not model.is_available():
                raise RuntimeError(f"Model {model_name} is not available")
            
            # Check total conversation length
            total_text = " ".join([msg.get("content", "") for msg in messages])
            if not model.check_input_length(total_text):
                self.logger.warning(f"Conversation may be too long for {model_name}")
                # Truncate older messages if needed
                messages = self._truncate_conversation(messages, model)
            
            # Generate response
            response, inference_time = model.chat(messages, params)
            
            # Validate output
            if not response or not isinstance(response, str):
                raise ValueError("Model returned empty or invalid response")
            
            # Analyze response quality
            quality_metrics = self._analyze_response_quality(messages[-1]["content"], response)
            
            result = {
                "model": model_name,
                "output": clean_response_text(response.strip()),
                "inference_time": inference_time,
                "success": True,
                "word_count": len(response.split()),
                "character_count": len(response),
                "quality_metrics": quality_metrics
            }
            
            self.logger.info(
                f"✓ {model_name}: {result['word_count']} words in {inference_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"✗ {model_name} failed: {error_msg}")
            
            return {
                "model": model_name,
                "output": None,
                "inference_time": total_time,
                "success": False,
                "error": error_msg,
                "error_type": type(e).__name__
            }

    def _clean_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Clean and validate conversation messages
        
        Args:
            messages: Raw message list
            
        Returns:
            Cleaned message list
        """
        cleaned = []
        
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            
            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "").strip()
            
            # Validate role
            if role not in ["system", "user", "assistant"]:
                role = "user"
            
            # Skip empty messages
            if not content:
                continue
            
            # Limit message length
            if len(content) > 10000:  # 10k character limit per message
                content = content[:10000] + "..."
                self.logger.warning("Message truncated due to length")
            
            cleaned.append({
                "role": role,
                "content": content
            })
        
        # Ensure we have at least one message
        if not cleaned:
            raise ValueError("No valid messages found")
        
        # Add system message if not present
        if cleaned[0]["role"] != "system":
            system_msg = {
                "role": "system",
                "content": "You are a helpful, knowledgeable, and friendly AI assistant. Provide clear, accurate, and helpful responses."
            }
            cleaned.insert(0, system_msg)
        
        return cleaned

    def _truncate_conversation(self, messages: List[Dict], model: BaseModel, max_length: int = 8000) -> List[Dict]:
        """
        Truncate conversation to fit within model limits
        
        Args:
            messages: Message list
            model: Model instance
            max_length: Maximum total character length
            
        Returns:
            Truncated message list
        """
        try:
            # Always keep system message and last user message
            if len(messages) <= 2:
                return messages
            
            system_msg = messages[0] if messages[0]["role"] == "system" else None
            last_msg = messages[-1]
            
            # Calculate current length
            total_length = sum(len(msg["content"]) for msg in messages)
            
            if total_length <= max_length:
                return messages
            
            # Build truncated conversation
            truncated = []
            if system_msg:
                truncated.append(system_msg)
            
            # Add recent messages working backwards
            current_length = len(system_msg["content"]) if system_msg else 0
            current_length += len(last_msg["content"])
            
            for msg in reversed(messages[1:-1]):  # Skip system and last message
                msg_length = len(msg["content"])
                if current_length + msg_length > max_length:
                    break
                truncated.insert(-1 if system_msg else 0, msg)
                current_length += msg_length
            
            truncated.append(last_msg)
            
            self.logger.info(f"Conversation truncated from {len(messages)} to {len(truncated)} messages")
            return truncated
            
        except Exception as e:
            self.logger.error(f"Error truncating conversation: {e}")
            return messages[-10:]  # Fallback: keep last 10 messages

    def _analyze_response_quality(self, user_input: str, response: str) -> Dict[str, Any]:
        """
        Analyze the quality of a chat response
        
        Args:
            user_input: User's input message
            response: Model's response
            
        Returns:
            Quality metrics
        """
        try:
            metrics = {
                "response_length": len(response),
                "word_count": len(response.split()),
                "sentence_count": len([s for s in response.split('.') if s.strip()]),
                "issues": []
            }
            
            # Check for common issues
            if len(response) < 10:
                metrics["issues"].append("Very short response")
            elif len(response) > 2000:
                metrics["issues"].append("Very long response")
            
            if response.lower() == user_input.lower():
                metrics["issues"].append("Response identical to input")
            
            # Check for repetition
            words = response.lower().split()
            if len(words) > 5:
                unique_words = set(words)
                repetition_ratio = 1 - (len(unique_words) / len(words))
                if repetition_ratio > 0.5:
                    metrics["issues"].append("High word repetition")
            
            # Check for coherence indicators
            if response.count("...") > 3:
                metrics["issues"].append("Many ellipses (may indicate uncertainty)")
            
            if response.count("I don't know") > 1:
                metrics["issues"].append("Multiple uncertainty statements")
            
            # Calculate overall quality score
            base_score = 1.0
            base_score -= len(metrics["issues"]) * 0.2
            
            # Length bonus/penalty
            if 50 <= len(response) <= 500:
                base_score += 0.1
            elif len(response) < 20 or len(response) > 1000:
                base_score -= 0.2
            
            metrics["quality_score"] = max(0.0, min(1.0, base_score))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing response quality: {e}")
            return {"error": str(e)}

    def _update_conversation_history(self, messages: List[Dict], results: List[Dict]):
        """Update conversation history for tracking"""
        try:
            conversation_id = f"conv_{int(time.time())}"
            
            self.conversation_history[conversation_id] = {
                "timestamp": time.time(),
                "messages": messages,
                "results": results,
                "successful_models": len([r for r in results if r.get("success")])
            }
            
            # Keep only last 10 conversations
            if len(self.conversation_history) > 10:
                oldest_key = min(self.conversation_history.keys(), 
                               key=lambda k: self.conversation_history[k]["timestamp"])
                del self.conversation_history[oldest_key]
                
        except Exception as e:
            self.logger.debug(f"Error updating conversation history: {e}")

    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return [model.get_name() for model in self.models if model.is_available()]

    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about all models"""
        return [model.get_model_info() for model in self.models]

    def chat_with_specific_model(self, messages: List[Dict], model_name: str, params: Dict[str, Any]) -> Dict:
        """
        Generate chat response with a specific model
        
        Args:
            messages: List of conversation messages
            model_name: Name of the model to use
            params: Generation parameters
            
        Returns:
            Result dictionary
        """
        try:
            # Find the model
            target_model = None
            for model in self.models:
                if model.get_name() == model_name:
                    target_model = model
                    break
            
            if target_model is None:
                raise ValueError(f"Model {model_name} not found or not available")
            
            # Validate inputs
            if not messages or not messages[-1].get("content"):
                raise ValueError("Chat input cannot be empty")
            
            messages = self._clean_messages(messages)
            params = validate_model_params(params)
            
            # Generate response
            return self._chat_with_model(target_model, messages, params)
            
        except Exception as e:
            self.logger.error(f"Error in model-specific chat: {e}")
            raise

    def get_conversation_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation history
        
        Args:
            limit: Maximum number of conversations to return
            
        Returns:
            List of conversation summaries
        """
        try:
            # Sort by timestamp and get recent ones
            sorted_convs = sorted(
                self.conversation_history.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True
            )[:limit]
            
            summaries = []
            for conv_id, conv_data in sorted_convs:
                summary = {
                    "id": conv_id,
                    "timestamp": conv_data["timestamp"],
                    "message_count": len(conv_data["messages"]),
                    "successful_models": conv_data["successful_models"],
                    "last_message": conv_data["messages"][-1]["content"][:100] + "..." if conv_data["messages"] else ""
                }
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return []

    def clear_conversation_history(self):
        """Clear all conversation history"""
        try:
            self.conversation_history.clear()
            self.logger.info("Conversation history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing conversation history: {e}")

    def export_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Export a specific conversation
        
        Args:
            conversation_id: ID of conversation to export
            
        Returns:
            Conversation data or None if not found
        """
        try:
            if conversation_id in self.conversation_history:
                return self.conversation_history[conversation_id]
            return None
        except Exception as e:
            self.logger.error(f"Error exporting conversation: {e}")
            return None