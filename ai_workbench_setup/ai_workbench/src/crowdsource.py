import sqlite3
import json
from utils.logger import setup_logger
import yaml
from typing import Dict, List

class CrowdsourceManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.db_path = self.config["crowdsourcing"]["db_path"]
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data TEXT NOT NULL,
                        submitter TEXT,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                self.logger.info("Initialized crowdsourced dataset DB")
        except Exception as e:
            self.logger.error(f"Error initializing DB: {str(e)}")
            raise

    def submit_dataset(self, data: List[Dict], submitter: str = "anonymous") -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO datasets (data, submitter, status) VALUES (?, ?, ?)",
                    (json.dumps(data), submitter, "pending")
                )
                conn.commit()
                dataset_id = cursor.lastrowid
                self.logger.info(f"Submitted dataset ID {dataset_id} by {submitter}")
                return dataset_id
        except Exception as e:
            self.logger.error(f"Error submitting dataset: {str(e)}")
            raise

    def get_pending_datasets(self) -> List[Dict]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, data, submitter FROM datasets WHERE status = 'pending'")
                rows = cursor.fetchall()
                return [{"id": row[0], "data": json.loads(row[1]), "submitter": row[2]} for row in rows]
        except Exception as e:
            self.logger.error(f"Error fetching pending datasets: {str(e)}")
            raise

    def approve_dataset(self, dataset_id: int):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE datasets SET status = 'approved' WHERE id = ?", (dataset_id,))
                conn.commit()
                self.logger.info(f"Approved dataset ID {dataset_id}")
        except Exception as e:
            self.logger.error(f"Error approving dataset: {str(e)}")
            raise