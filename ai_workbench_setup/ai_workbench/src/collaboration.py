from utils.logger import setup_logger
import yaml
from typing import List, Dict

class CollaborationArena:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.websocket_url = f"ws://{self.config['websocket']['host']}:{self.config['websocket']['port']}/ws"
        self.logger.info(f"CollaborationArena initialized with WebSocket URL: {self.websocket_url}")

    async def broadcast_results(self, task: str, results: List[Dict]):
        try:
            # Temporarily skip WebSocket broadcasting to avoid errors
            self.logger.info(f"Skipping WebSocket broadcast for task: {task}, results: {results}")
            # Placeholder for future WebSocket implementation
            pass
        except Exception as e:
            self.logger.error(f"Error broadcasting to arena: {str(e)}")
            raise

    async def subscribe_to_arena(self, callback):
        try:
            # Temporarily skip WebSocket subscription
            self.logger.info("Skipping WebSocket subscription")
            # Placeholder for future WebSocket implementation
            pass
        except Exception as e:
            self.logger.error(f"Error subscribing to arena: {str(e)}")
            raise