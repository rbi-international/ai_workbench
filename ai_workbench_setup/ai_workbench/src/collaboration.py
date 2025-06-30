from utils.logger import setup_logger
import yaml
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Callable
from collections import defaultdict
import uuid

class CollaborationArena:
    """
    Enhanced collaboration arena for real-time model battles, comparisons,
    and collaborative AI evaluation with WebSocket support
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            websocket_config = config.get("websocket", {})
            self.host = websocket_config.get("host", "127.0.0.1")
            self.port = websocket_config.get("port", 8080)
            
        except Exception as e:
            self.logger.warning(f"Could not load collaboration configuration: {e}")
            self.host = "127.0.0.1"
            self.port = 8080
        
        # Arena state
        self.active_battles = {}  # battle_id -> battle_data
        self.connected_clients = {}  # client_id -> client_info
        self.battle_history = []  # Historical battle results
        self.leaderboard = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "total_battles": 0, "score": 0.0})
        self.event_listeners = {}  # event_type -> [callbacks]
        
        # WebSocket connections (placeholder - actual implementation would use FastAPI WebSocket)
        self.websocket_connections = set()
        
        # Battle configuration
        self.max_battle_duration = 300  # 5 minutes
        self.min_participants = 2
        self.max_participants = 10
        
        self.logger.info(f"Collaboration arena initialized on {self.host}:{self.port}")

    async def create_battle(self, battle_config: Dict[str, Any], creator_id: str) -> str:
        """
        Create a new model battle
        
        Args:
            battle_config: Battle configuration (task, models, parameters, etc.)
            creator_id: ID of the battle creator
            
        Returns:
            Battle ID
        """
        try:
            battle_id = str(uuid.uuid4())
            
            # Validate battle configuration
            validation_result = self._validate_battle_config(battle_config)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid battle config: {validation_result['errors']}")
            
            # Create battle data structure
            battle_data = {
                "id": battle_id,
                "creator_id": creator_id,
                "config": battle_config,
                "status": "waiting",  # waiting, active, completed, cancelled
                "participants": {creator_id: {"role": "creator", "joined_at": datetime.now().isoformat()}},
                "models": battle_config.get("models", []),
                "task": battle_config.get("task", "general"),
                "input_text": battle_config.get("input_text", ""),
                "results": {},
                "votes": {},
                "scores": {},
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "winner": None,
                "metadata": battle_config.get("metadata", {})
            }
            
            self.active_battles[battle_id] = battle_data
            
            # Notify connected clients
            await self._broadcast_event("battle_created", {
                "battle_id": battle_id,
                "creator": creator_id,
                "task": battle_data["task"],
                "models": battle_data["models"]
            })
            
            self.logger.info(f"Battle {battle_id} created by {creator_id}")
            return battle_id
            
        except Exception as e:
            self.logger.error(f"Error creating battle: {e}")
            raise

    def _validate_battle_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate battle configuration"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["task", "models", "input_text"]
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate models
        models = config.get("models", [])
        if not isinstance(models, list) or len(models) < 2:
            errors.append("At least 2 models are required for a battle")
        
        if len(models) > 5:
            warnings.append("Large number of models may slow down the battle")
        
        # Validate task
        valid_tasks = ["summarization", "translation", "chat", "general"]
        if config.get("task") not in valid_tasks:
            errors.append(f"Invalid task. Must be one of: {valid_tasks}")
        
        # Validate input text
        input_text = config.get("input_text", "")
        if not input_text or len(input_text.strip()) < 10:
            errors.append("Input text must be at least 10 characters")
        
        if len(input_text) > 10000:
            warnings.append("Very long input text may affect performance")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

    async def join_battle(self, battle_id: str, user_id: str, role: str = "participant") -> bool:
        """
        Join an existing battle
        
        Args:
            battle_id: Battle ID
            user_id: User ID
            role: User role (participant, observer)
            
        Returns:
            Success status
        """
        try:
            if battle_id not in self.active_battles:
                raise ValueError(f"Battle {battle_id} not found")
            
            battle = self.active_battles[battle_id]
            
            if battle["status"] != "waiting":
                raise ValueError(f"Battle {battle_id} is not accepting new participants (status: {battle['status']})")
            
            if len(battle["participants"]) >= self.max_participants:
                raise ValueError(f"Battle {battle_id} is full")
            
            if user_id in battle["participants"]:
                self.logger.warning(f"User {user_id} already in battle {battle_id}")
                return True
            
            # Add participant
            battle["participants"][user_id] = {
                "role": role,
                "joined_at": datetime.now().isoformat()
            }
            
            # Start battle if minimum participants reached
            if len(battle["participants"]) >= self.min_participants and battle["status"] == "waiting":
                await self._start_battle(battle_id)
            
            # Notify participants
            await self._broadcast_to_battle(battle_id, "participant_joined", {
                "user_id": user_id,
                "role": role,
                "participant_count": len(battle["participants"])
            })
            
            self.logger.info(f"User {user_id} joined battle {battle_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error joining battle {battle_id}: {e}")
            return False

    async def _start_battle(self, battle_id: str):
        """Start a battle"""
        try:
            battle = self.active_battles[battle_id]
            
            if battle["status"] != "waiting":
                return
            
            battle["status"] = "active"
            battle["started_at"] = datetime.now().isoformat()
            
            # Generate results for each model (simulate API calls)
            await self._generate_battle_results(battle_id)
            
            # Notify participants
            await self._broadcast_to_battle(battle_id, "battle_started", {
                "battle_id": battle_id,
                "models": battle["models"],
                "task": battle["task"]
            })
            
            self.logger.info(f"Battle {battle_id} started")
            
        except Exception as e:
            self.logger.error(f"Error starting battle {battle_id}: {e}")

    async def _generate_battle_results(self, battle_id: str):
        """Generate results for battle models (placeholder for actual AI processing)"""
        try:
            battle = self.active_battles[battle_id]
            
            # This would integrate with your actual AI models
            # For now, we'll simulate results
            for model in battle["models"]:
                # Simulate processing time
                await asyncio.sleep(0.1)
                
                # Generate mock result
                result = {
                    "model": model,
                    "output": f"Mock output from {model} for task: {battle['task']}",
                    "inference_time": 0.5 + (hash(model) % 100) / 100,  # Mock time
                    "quality_score": 0.7 + (hash(model) % 30) / 100,  # Mock quality
                    "generated_at": datetime.now().isoformat()
                }
                
                battle["results"][model] = result
            
            # Notify participants that results are ready
            await self._broadcast_to_battle(battle_id, "results_ready", {
                "battle_id": battle_id,
                "models": list(battle["results"].keys())
            })
            
        except Exception as e:
            self.logger.error(f"Error generating battle results: {e}")

    async def submit_vote(self, battle_id: str, user_id: str, votes: Dict[str, float]) -> bool:
        """
        Submit votes for battle results
        
        Args:
            battle_id: Battle ID
            user_id: Voter ID
            votes: Dictionary of model -> score votes
            
        Returns:
            Success status
        """
        try:
            if battle_id not in self.active_battles:
                raise ValueError(f"Battle {battle_id} not found")
            
            battle = self.active_battles[battle_id]
            
            if battle["status"] != "active":
                raise ValueError(f"Battle {battle_id} is not active for voting")
            
            if user_id not in battle["participants"]:
                raise ValueError(f"User {user_id} is not a participant in battle {battle_id}")
            
            # Validate votes
            for model, score in votes.items():
                if model not in battle["results"]:
                    raise ValueError(f"Model {model} not in battle")
                
                if not isinstance(score, (int, float)) or score < 0 or score > 10:
                    raise ValueError(f"Score for {model} must be between 0 and 10")
            
            # Store votes
            battle["votes"][user_id] = {
                "votes": votes,
                "submitted_at": datetime.now().isoformat()
            }
            
            # Check if all participants have voted
            participant_count = len([p for p in battle["participants"].values() if p["role"] == "participant"])
            vote_count = len(battle["votes"])
            
            if vote_count >= participant_count:
                await self._complete_battle(battle_id)
            
            # Notify participants
            await self._broadcast_to_battle(battle_id, "vote_submitted", {
                "user_id": user_id,
                "votes_received": vote_count,
                "total_participants": participant_count
            })
            
            self.logger.info(f"Vote submitted by {user_id} for battle {battle_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting vote: {e}")
            return False

    async def _complete_battle(self, battle_id: str):
        """Complete a battle and calculate results"""
        try:
            battle = self.active_battles[battle_id]
            
            if battle["status"] != "active":
                return
            
            battle["status"] = "completed"
            battle["completed_at"] = datetime.now().isoformat()
            
            # Calculate final scores
            final_scores = self._calculate_final_scores(battle)
            battle["scores"] = final_scores
            
            # Determine winner
            if final_scores:
                winner = max(final_scores, key=final_scores.get)
                battle["winner"] = winner
            
            # Update leaderboard
            self._update_leaderboard(battle)
            
            # Move to history
            self.battle_history.append(dict(battle))
            
            # Notify participants
            await self._broadcast_to_battle(battle_id, "battle_completed", {
                "battle_id": battle_id,
                "winner": battle.get("winner"),
                "scores": final_scores
            })
            
            self.logger.info(f"Battle {battle_id} completed, winner: {battle.get('winner')}")
            
        except Exception as e:
            self.logger.error(f"Error completing battle {battle_id}: {e}")

    def _calculate_final_scores(self, battle: Dict) -> Dict[str, float]:
        """Calculate final scores from votes"""
        try:
            model_scores = defaultdict(list)
            
            # Collect all votes for each model
            for user_votes in battle["votes"].values():
                for model, score in user_votes["votes"].items():
                    model_scores[model].append(score)
            
            # Calculate average scores
            final_scores = {}
            for model, scores in model_scores.items():
                if scores:
                    final_scores[model] = sum(scores) / len(scores)
                else:
                    final_scores[model] = 0.0
            
            return final_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating final scores: {e}")
            return {}

    def _update_leaderboard(self, battle: Dict):
        """Update the global leaderboard"""
        try:
            scores = battle.get("scores", {})
            if not scores:
                return
            
            # Sort models by score
            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_models) >= 2:
                winner = sorted_models[0][0]
                winner_score = sorted_models[0][1]
                
                # Update stats for all models
                for i, (model, score) in enumerate(sorted_models):
                    self.leaderboard[model]["total_battles"] += 1
                    
                    if i == 0 and len(sorted_models) > 1 and winner_score > sorted_models[1][1]:
                        # Clear winner
                        self.leaderboard[model]["wins"] += 1
                    elif i == len(sorted_models) - 1 and winner_score > score:
                        # Clear loser
                        self.leaderboard[model]["losses"] += 1
                    else:
                        # Draw or close scores
                        self.leaderboard[model]["draws"] += 1
                
                # Recalculate ELO-style scores
                self._recalculate_elo_scores(sorted_models)
        
        except Exception as e:
            self.logger.error(f"Error updating leaderboard: {e}")

    def _recalculate_elo_scores(self, sorted_models: List[Tuple[str, float]]):
        """Recalculate ELO-style scores for models"""
        try:
            k_factor = 32  # ELO K-factor
            
            for i, (model_a, score_a) in enumerate(sorted_models):
                for j, (model_b, score_b) in enumerate(sorted_models):
                    if i != j:
                        # Calculate expected score
                        rating_a = self.leaderboard[model_a]["score"]
                        rating_b = self.leaderboard[model_b]["score"]
                        
                        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                        
                        # Actual score (1 for win, 0.5 for draw, 0 for loss)
                        if score_a > score_b:
                            actual_a = 1.0
                        elif score_a == score_b:
                            actual_a = 0.5
                        else:
                            actual_a = 0.0
                        
                        # Update rating
                        new_rating = rating_a + k_factor * (actual_a - expected_a)
                        self.leaderboard[model_a]["score"] = new_rating
        
        except Exception as e:
            self.logger.error(f"Error recalculating ELO scores: {e}")

    async def get_battle_status(self, battle_id: str) -> Optional[Dict]:
        """Get current battle status"""
        try:
            if battle_id in self.active_battles:
                battle = self.active_battles[battle_id]
                
                # Create safe copy without sensitive data
                status = {
                    "id": battle["id"],
                    "status": battle["status"],
                    "task": battle["task"],
                    "models": battle["models"],
                    "participant_count": len(battle["participants"]),
                    "vote_count": len(battle["votes"]),
                    "created_at": battle["created_at"],
                    "started_at": battle.get("started_at"),
                    "completed_at": battle.get("completed_at")
                }
                
                if battle["status"] == "completed":
                    status["winner"] = battle.get("winner")
                    status["scores"] = battle.get("scores", {})
                
                return status
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting battle status: {e}")
            return None

    async def get_leaderboard(self, limit: int = 20) -> List[Dict]:
        """Get current leaderboard"""
        try:
            # Sort by ELO score
            sorted_models = sorted(
                self.leaderboard.items(),
                key=lambda x: x[1]["score"],
                reverse=True
            )
            
            leaderboard = []
            for i, (model, stats) in enumerate(sorted_models[:limit]):
                win_rate = stats["wins"] / max(stats["total_battles"], 1)
                
                leaderboard.append({
                    "rank": i + 1,
                    "model": model,
                    "elo_score": round(stats["score"], 2),
                    "total_battles": stats["total_battles"],
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "draws": stats["draws"],
                    "win_rate": round(win_rate, 3)
                })
            
            return leaderboard
            
        except Exception as e:
            self.logger.error(f"Error getting leaderboard: {e}")
            return []

    async def get_battle_history(self, limit: int = 50, user_id: str = None) -> List[Dict]:
        """Get battle history"""
        try:
            history = self.battle_history[-limit:] if limit else self.battle_history
            
            if user_id:
                # Filter battles where user participated
                history = [
                    battle for battle in history
                    if user_id in battle.get("participants", {})
                ]
            
            # Return safe summary
            summary = []
            for battle in history:
                summary.append({
                    "id": battle["id"],
                    "task": battle["task"],
                    "models": battle["models"],
                    "winner": battle.get("winner"),
                    "participant_count": len(battle.get("participants", {})),
                    "completed_at": battle.get("completed_at"),
                    "scores": battle.get("scores", {})
                })
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting battle history: {e}")
            return []

    async def cancel_battle(self, battle_id: str, user_id: str) -> bool:
        """Cancel a battle"""
        try:
            if battle_id not in self.active_battles:
                return False
            
            battle = self.active_battles[battle_id]
            
            # Only creator can cancel
            if battle["creator_id"] != user_id:
                return False
            
            if battle["status"] not in ["waiting", "active"]:
                return False
            
            battle["status"] = "cancelled"
            battle["completed_at"] = datetime.now().isoformat()
            
            # Notify participants
            await self._broadcast_to_battle(battle_id, "battle_cancelled", {
                "battle_id": battle_id,
                "cancelled_by": user_id
            })
            
            # Move to history
            self.battle_history.append(dict(battle))
            del self.active_battles[battle_id]
            
            self.logger.info(f"Battle {battle_id} cancelled by {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling battle: {e}")
            return False

    async def _broadcast_event(self, event_type: str, data: Dict):
        """Broadcast event to all connected clients"""
        try:
            message = {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # In a real implementation, this would send via WebSocket
            self.logger.debug(f"Broadcasting {event_type}: {data}")
            
            # Call registered event listeners
            if event_type in self.event_listeners:
                for callback in self.event_listeners[event_type]:
                    try:
                        await callback(data)
                    except Exception as e:
                        self.logger.error(f"Error in event listener: {e}")
            
        except Exception as e:
            self.logger.error(f"Error broadcasting event: {e}")

    async def _broadcast_to_battle(self, battle_id: str, event_type: str, data: Dict):
        """Broadcast event to battle participants"""
        try:
            if battle_id not in self.active_battles:
                return
            
            battle = self.active_battles[battle_id]
            participants = list(battle["participants"].keys())
            
            message = {
                "type": event_type,
                "battle_id": battle_id,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # In a real implementation, this would send to specific WebSocket connections
            self.logger.debug(f"Broadcasting to battle {battle_id} participants {participants}: {event_type}")
            
        except Exception as e:
            self.logger.error(f"Error broadcasting to battle: {e}")

    def register_event_listener(self, event_type: str, callback: Callable):
        """Register event listener"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        
        self.event_listeners[event_type].append(callback)
        self.logger.debug(f"Registered listener for {event_type}")

    async def broadcast_results(self, task: str, results: List[Dict]):
        """
        Broadcast AI processing results (legacy method for compatibility)
        
        Args:
            task: Task type
            results: Processing results
        """
        try:
            await self._broadcast_event("ai_results", {
                "task": task,
                "results": results
            })
            
        except Exception as e:
            self.logger.error(f"Error broadcasting results: {e}")

    def get_arena_statistics(self) -> Dict[str, Any]:
        """Get arena statistics"""
        try:
            total_battles = len(self.battle_history) + len(self.active_battles)
            active_battles = len([b for b in self.active_battles.values() if b["status"] == "active"])
            
            # Calculate average battle duration
            completed_battles = [b for b in self.battle_history if b.get("completed_at") and b.get("started_at")]
            avg_duration = 0
            
            if completed_battles:
                durations = []
                for battle in completed_battles:
                    start = datetime.fromisoformat(battle["started_at"])
                    end = datetime.fromisoformat(battle["completed_at"])
                    duration = (end - start).total_seconds()
                    durations.append(duration)
                
                avg_duration = sum(durations) / len(durations)
            
            # Most popular models
            model_usage = defaultdict(int)
            for battle in self.battle_history:
                for model in battle.get("models", []):
                    model_usage[model] += 1
            
            popular_models = sorted(model_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "total_battles": total_battles,
                "active_battles": active_battles,
                "completed_battles": len(self.battle_history),
                "average_battle_duration_seconds": round(avg_duration, 2),
                "total_models": len(self.leaderboard),
                "popular_models": [{"model": model, "usage_count": count} for model, count in popular_models],
                "leaderboard_top_3": (await self.get_leaderboard(3))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting arena statistics: {e}")
            return {"error": str(e)}

    async def cleanup_old_battles(self, hours_old: int = 24):
        """Clean up old completed battles"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_old)
            
            battles_to_remove = []
            for battle_id, battle in self.active_battles.items():
                if battle["status"] in ["completed", "cancelled"]:
                    completed_time = battle.get("completed_at")
                    if completed_time:
                        completed_dt = datetime.fromisoformat(completed_time)
                        if completed_dt < cutoff_time:
                            battles_to_remove.append(battle_id)
            
            # Move old battles to history and remove from active
            for battle_id in battles_to_remove:
                battle = self.active_battles[battle_id]
                if battle not in self.battle_history:
                    self.battle_history.append(dict(battle))
                del self.active_battles[battle_id]
            
            # Limit history size
            max_history = 1000
            if len(self.battle_history) > max_history:
                self.battle_history = self.battle_history[-max_history:]
            
            self.logger.info(f"Cleaned up {len(battles_to_remove)} old battles")
            return len(battles_to_remove)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old battles: {e}")
            return 0