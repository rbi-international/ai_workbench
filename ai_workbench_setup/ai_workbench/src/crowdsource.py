import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from utils.logger import setup_logger
import yaml

class CrowdsourceManager:
    """
    Enhanced crowdsourcing manager for collecting, validating, and managing
    user-contributed datasets with quality control and moderation features
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger(__name__)
        
        # Load configuration
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            crowdsource_config = config.get("crowdsourcing", {})
            self.db_path = crowdsource_config.get("db_path", "data/crowdsourced/datasets.db")
            self.enabled = crowdsource_config.get("enabled", True)
            
        except Exception as e:
            self.logger.warning(f"Could not load crowdsourcing configuration: {e}")
            self.db_path = "data/crowdsourced/datasets.db"
            self.enabled = True
        
        # Ensure database directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Quality control settings
        self.min_data_points = 1
        self.max_data_points = 1000
        self.max_text_length = 10000
        self.prohibited_content = self._load_prohibited_content()
        
        self.logger.info(f"Crowdsource manager initialized with database: {self.db_path}")

    def _init_database(self):
        """Initialize the crowdsourcing database with comprehensive schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Main datasets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS datasets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data_hash TEXT UNIQUE NOT NULL,
                        data TEXT NOT NULL,
                        submitter TEXT NOT NULL,
                        submitter_email TEXT,
                        submission_ip TEXT,
                        status TEXT DEFAULT 'pending',
                        priority INTEGER DEFAULT 0,
                        category TEXT,
                        task_type TEXT,
                        quality_score REAL DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        reviewed_at TIMESTAMP,
                        reviewer TEXT,
                        review_notes TEXT,
                        approval_reason TEXT,
                        rejection_reason TEXT,
                        metadata TEXT
                    )
                """)
                
                # Submitter tracking table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS submitters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        submitter_name TEXT UNIQUE NOT NULL,
                        email TEXT,
                        total_submissions INTEGER DEFAULT 0,
                        approved_submissions INTEGER DEFAULT 0,
                        rejected_submissions INTEGER DEFAULT 0,
                        reputation_score REAL DEFAULT 0.0,
                        first_submission TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_submission TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_trusted BOOLEAN DEFAULT FALSE,
                        is_blocked BOOLEAN DEFAULT FALSE,
                        notes TEXT
                    )
                """)
                
                # Quality metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_id INTEGER,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                    )
                """)
                
                # Feedback table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        dataset_id INTEGER,
                        feedback_type TEXT NOT NULL,
                        feedback_text TEXT,
                        rating INTEGER,
                        reviewer TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (dataset_id) REFERENCES datasets (id)
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_submitter ON datasets(submitter)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_submitters_name ON submitters(submitter_name)")
                
                conn.commit()
                self.logger.info("Database schema initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def _load_prohibited_content(self) -> List[str]:
        """Load patterns for prohibited content"""
        return [
            # Personal information
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
            r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card
            
            # Harmful content
            r'\b(password|login|secret|confidential)\b',
            r'\b(kill|murder|harm|violence|threat)\b',
            r'\b(hate|racist|sexist|bigot)\b',
            
            # Spam indicators
            r'\b(click here|buy now|limited time|act fast)\b',
            r'\b(viagra|casino|lottery|inheritance)\b'
        ]

    def submit_dataset(self, data: List[Dict], submitter: str, 
                      submitter_email: str = None, task_type: str = None, 
                      category: str = None, metadata: Dict = None) -> int:
        """
        Submit a new dataset for review
        
        Args:
            data: List of data points
            submitter: Name/ID of submitter
            submitter_email: Optional email address
            task_type: Type of task (summarization, translation, etc.)
            category: Dataset category
            metadata: Additional metadata
            
        Returns:
            Dataset ID if successful
        """
        if not self.enabled:
            raise RuntimeError("Crowdsourcing is disabled")
        
        try:
            # Validate input
            validation_result = self._validate_dataset(data, submitter)
            if not validation_result["valid"]:
                raise ValueError(f"Dataset validation failed: {validation_result['errors']}")
            
            # Create data hash for deduplication
            data_str = json.dumps(data, sort_keys=True)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()
            
            # Calculate initial quality score
            quality_score = self._calculate_quality_score(data)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "data_points_count": len(data),
                "submission_timestamp": datetime.now().isoformat(),
                "validation_score": validation_result.get("score", 0.0)
            })
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for duplicate
                cursor.execute("SELECT id FROM datasets WHERE data_hash = ?", (data_hash,))
                existing = cursor.fetchone()
                if existing:
                    self.logger.warning(f"Duplicate dataset submission from {submitter}")
                    raise ValueError("Dataset already exists")
                
                # Insert dataset
                cursor.execute("""
                    INSERT INTO datasets 
                    (data_hash, data, submitter, submitter_email, status, category, 
                     task_type, quality_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data_hash, data_str, submitter, submitter_email, "pending",
                    category, task_type, quality_score, json.dumps(metadata)
                ))
                
                dataset_id = cursor.lastrowid
                
                # Update submitter statistics
                self._update_submitter_stats(cursor, submitter, submitter_email, "submission")
                
                conn.commit()
                
                self.logger.info(f"Dataset {dataset_id} submitted by {submitter} (quality: {quality_score:.2f})")
                return dataset_id
                
        except Exception as e:
            self.logger.error(f"Error submitting dataset: {e}")
            raise

    def _validate_dataset(self, data: List[Dict], submitter: str) -> Dict[str, Any]:
        """Validate submitted dataset"""
        errors = []
        warnings = []
        score = 1.0
        
        # Check data structure
        if not isinstance(data, list):
            errors.append("Data must be a list")
            return {"valid": False, "errors": errors}
        
        if len(data) < self.min_data_points:
            errors.append(f"Dataset must have at least {self.min_data_points} data points")
        
        if len(data) > self.max_data_points:
            errors.append(f"Dataset cannot exceed {self.max_data_points} data points")
        
        # Validate individual data points
        for i, item in enumerate(data[:10]):  # Check first 10 items
            if not isinstance(item, dict):
                errors.append(f"Data point {i} must be a dictionary")
                continue
            
            # Check for required fields (flexible)
            if not any(key in item for key in ["text", "input", "content", "prompt"]):
                warnings.append(f"Data point {i} missing common text fields")
                score -= 0.1
            
            # Check text length
            for key, value in item.items():
                if isinstance(value, str) and len(value) > self.max_text_length:
                    warnings.append(f"Text in {key} exceeds maximum length")
                    score -= 0.1
        
        # Check for prohibited content
        prohibited_count = self._check_prohibited_content(data)
        if prohibited_count > 0:
            errors.append(f"Dataset contains {prohibited_count} instances of prohibited content")
            score -= prohibited_count * 0.2
        
        # Check submitter reputation
        submitter_reputation = self._get_submitter_reputation(submitter)
        if submitter_reputation < 0.3:
            warnings.append("Submitter has low reputation score")
            score -= 0.2
        
        is_valid = len(errors) == 0
        
        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "score": max(0.0, min(1.0, score))
        }

    def _check_prohibited_content(self, data: List[Dict]) -> int:
        """Check for prohibited content patterns"""
        import re
        
        prohibited_count = 0
        
        for item in data:
            text_content = json.dumps(item).lower()
            
            for pattern in self.prohibited_content:
                if re.search(pattern, text_content, re.IGNORECASE):
                    prohibited_count += 1
        
        return prohibited_count

    def _calculate_quality_score(self, data: List[Dict]) -> float:
        """Calculate initial quality score for dataset"""
        score = 0.5  # Base score
        
        # Size factor
        size_factor = min(1.0, len(data) / 50.0)  # Optimal around 50 items
        score += size_factor * 0.3
        
        # Diversity factor
        unique_items = len(set(json.dumps(item, sort_keys=True) for item in data))
        diversity_factor = unique_items / len(data) if len(data) > 0 else 0
        score += diversity_factor * 0.2
        
        # Completeness factor
        complete_items = 0
        for item in data:
            if isinstance(item, dict) and len(item) >= 2:  # At least 2 fields
                complete_items += 1
        
        completeness = complete_items / len(data) if len(data) > 0 else 0
        score += completeness * 0.3
        
        return min(1.0, max(0.0, score))

    def _get_submitter_reputation(self, submitter: str) -> float:
        """Get submitter's reputation score"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT reputation_score, is_trusted, is_blocked 
                    FROM submitters WHERE submitter_name = ?
                """, (submitter,))
                
                result = cursor.fetchone()
                if result:
                    reputation, is_trusted, is_blocked = result
                    if is_blocked:
                        return 0.0
                    if is_trusted:
                        return min(1.0, reputation + 0.2)
                    return reputation
                else:
                    return 0.5  # Default for new submitters
        except Exception as e:
            self.logger.error(f"Error getting submitter reputation: {e}")
            return 0.5

    def _update_submitter_stats(self, cursor, submitter: str, email: str, action: str):
        """Update submitter statistics"""
        try:
            # Check if submitter exists
            cursor.execute("SELECT id FROM submitters WHERE submitter_name = ?", (submitter,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing submitter
                if action == "submission":
                    cursor.execute("""
                        UPDATE submitters 
                        SET total_submissions = total_submissions + 1,
                            last_submission = CURRENT_TIMESTAMP
                        WHERE submitter_name = ?
                    """, (submitter,))
                elif action == "approval":
                    cursor.execute("""
                        UPDATE submitters 
                        SET approved_submissions = approved_submissions + 1
                        WHERE submitter_name = ?
                    """, (submitter,))
                elif action == "rejection":
                    cursor.execute("""
                        UPDATE submitters 
                        SET rejected_submissions = rejected_submissions + 1
                        WHERE submitter_name = ?
                    """, (submitter,))
            else:
                # Create new submitter
                cursor.execute("""
                    INSERT INTO submitters 
                    (submitter_name, email, total_submissions, reputation_score)
                    VALUES (?, ?, 1, 0.5)
                """, (submitter, email))
            
            # Recalculate reputation
            self._recalculate_reputation(cursor, submitter)
            
        except Exception as e:
            self.logger.error(f"Error updating submitter stats: {e}")

    def _recalculate_reputation(self, cursor, submitter: str):
        """Recalculate submitter reputation score"""
        try:
            cursor.execute("""
                SELECT total_submissions, approved_submissions, rejected_submissions
                FROM submitters WHERE submitter_name = ?
            """, (submitter,))
            
            result = cursor.fetchone()
            if result:
                total, approved, rejected = result
                
                if total > 0:
                    approval_rate = approved / total
                    rejection_rate = rejected / total
                    
                    # Base reputation on approval rate with adjustments
                    reputation = approval_rate * 0.8
                    
                    # Bonus for consistency
                    if total >= 10:
                        reputation += 0.1
                    
                    # Penalty for high rejection rate
                    if rejection_rate > 0.3:
                        reputation -= 0.2
                    
                    reputation = max(0.0, min(1.0, reputation))
                    
                    cursor.execute("""
                        UPDATE submitters 
                        SET reputation_score = ?
                        WHERE submitter_name = ?
                    """, (reputation, submitter))
                    
        except Exception as e:
            self.logger.error(f"Error recalculating reputation: {e}")

    def get_pending_datasets(self, limit: int = 50, priority_order: bool = True) -> List[Dict]:
        """
        Get pending datasets for moderation
        
        Args:
            limit: Maximum number of datasets to return
            priority_order: Whether to order by priority and quality
            
        Returns:
            List of pending datasets
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if priority_order:
                    order_clause = "ORDER BY priority DESC, quality_score DESC, created_at ASC"
                else:
                    order_clause = "ORDER BY created_at ASC"
                
                cursor.execute(f"""
                    SELECT d.id, d.data, d.submitter, d.submitter_email, d.category,
                           d.task_type, d.quality_score, d.created_at, d.metadata,
                           s.reputation_score, s.total_submissions, s.approved_submissions
                    FROM datasets d
                    LEFT JOIN submitters s ON d.submitter = s.submitter_name
                    WHERE d.status = 'pending'
                    {order_clause}
                    LIMIT ?
                """, (limit,))
                
                results = cursor.fetchall()
                
                datasets = []
                for row in results:
                    try:
                        metadata = json.loads(row[8]) if row[8] else {}
                        datasets.append({
                            "id": row[0],
                            "data": json.loads(row[1]),
                            "submitter": row[2],
                            "submitter_email": row[3],
                            "category": row[4],
                            "task_type": row[5],
                            "quality_score": row[6],
                            "created_at": row[7],
                            "metadata": metadata,
                            "submitter_reputation": row[9] or 0.5,
                            "submitter_total": row[10] or 0,
                            "submitter_approved": row[11] or 0
                        })
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing dataset {row[0]}: {e}")
                        continue
                
                return datasets
                
        except Exception as e:
            self.logger.error(f"Error fetching pending datasets: {e}")
            return []

    def approve_dataset(self, dataset_id: int, reviewer: str = "system", 
                       approval_reason: str = None, notes: str = None) -> bool:
        """
        Approve a dataset
        
        Args:
            dataset_id: ID of dataset to approve
            reviewer: Name of reviewer
            approval_reason: Reason for approval
            notes: Additional notes
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get dataset info
                cursor.execute("""
                    SELECT submitter, status FROM datasets WHERE id = ?
                """, (dataset_id,))
                
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Dataset {dataset_id} not found")
                
                submitter, current_status = result
                
                if current_status != "pending":
                    raise ValueError(f"Dataset {dataset_id} is not pending (status: {current_status})")
                
                # Update dataset status
                cursor.execute("""
                    UPDATE datasets 
                    SET status = 'approved', 
                        reviewed_at = CURRENT_TIMESTAMP,
                        reviewer = ?,
                        approval_reason = ?,
                        review_notes = ?
                    WHERE id = ?
                """, (reviewer, approval_reason, notes, dataset_id))
                
                # Update submitter stats
                self._update_submitter_stats(cursor, submitter, None, "approval")
                
                # Add feedback entry
                if notes:
                    cursor.execute("""
                        INSERT INTO feedback (dataset_id, feedback_type, feedback_text, reviewer)
                        VALUES (?, 'approval', ?, ?)
                    """, (dataset_id, notes, reviewer))
                
                conn.commit()
                
                self.logger.info(f"Dataset {dataset_id} approved by {reviewer}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error approving dataset {dataset_id}: {e}")
            return False

    def reject_dataset(self, dataset_id: int, reviewer: str = "system",
                      rejection_reason: str = None, notes: str = None) -> bool:
        """
        Reject a dataset
        
        Args:
            dataset_id: ID of dataset to reject
            reviewer: Name of reviewer
            rejection_reason: Reason for rejection
            notes: Additional notes
            
        Returns:
            Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get dataset info
                cursor.execute("""
                    SELECT submitter, status FROM datasets WHERE id = ?
                """, (dataset_id,))
                
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Dataset {dataset_id} not found")
                
                submitter, current_status = result
                
                if current_status != "pending":
                    raise ValueError(f"Dataset {dataset_id} is not pending (status: {current_status})")
                
                # Update dataset status
                cursor.execute("""
                    UPDATE datasets 
                    SET status = 'rejected',
                        reviewed_at = CURRENT_TIMESTAMP,
                        reviewer = ?,
                        rejection_reason = ?,
                        review_notes = ?
                    WHERE id = ?
                """, (reviewer, rejection_reason, notes, dataset_id))
                
                # Update submitter stats
                self._update_submitter_stats(cursor, submitter, None, "rejection")
                
                # Add feedback entry
                if notes:
                    cursor.execute("""
                        INSERT INTO feedback (dataset_id, feedback_type, feedback_text, reviewer)
                        VALUES (?, 'rejection', ?, ?)
                    """, (dataset_id, notes, reviewer))
                
                conn.commit()
                
                self.logger.info(f"Dataset {dataset_id} rejected by {reviewer}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error rejecting dataset {dataset_id}: {e}")
            return False

    def get_approved_datasets(self, task_type: str = None, category: str = None, 
                            limit: int = 100) -> List[Dict]:
        """
        Get approved datasets for use
        
        Args:
            task_type: Filter by task type
            category: Filter by category
            limit: Maximum number of datasets
            
        Returns:
            List of approved datasets
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT id, data, submitter, category, task_type, 
                           quality_score, created_at, reviewed_at
                    FROM datasets 
                    WHERE status = 'approved'
                """
                params = []
                
                if task_type:
                    query += " AND task_type = ?"
                    params.append(task_type)
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                query += " ORDER BY quality_score DESC, reviewed_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                datasets = []
                for row in results:
                    try:
                        datasets.append({
                            "id": row[0],
                            "data": json.loads(row[1]),
                            "submitter": row[2],
                            "category": row[3],
                            "task_type": row[4],
                            "quality_score": row[5],
                            "created_at": row[6],
                            "reviewed_at": row[7]
                        })
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Error parsing approved dataset {row[0]}: {e}")
                        continue
                
                return datasets
                
        except Exception as e:
            self.logger.error(f"Error fetching approved datasets: {e}")
            return []

    def get_submitter_statistics(self, submitter: str = None) -> Dict[str, Any]:
        """
        Get statistics for submitters
        
        Args:
            submitter: Specific submitter name, or None for all
            
        Returns:
            Statistics dictionary
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if submitter:
                    # Individual submitter stats
                    cursor.execute("""
                        SELECT submitter_name, total_submissions, approved_submissions,
                               rejected_submissions, reputation_score, is_trusted, is_blocked,
                               first_submission, last_submission
                        FROM submitters WHERE submitter_name = ?
                    """, (submitter,))
                    
                    result = cursor.fetchone()
                    if not result:
                        return {"error": f"Submitter {submitter} not found"}
                    
                    return {
                        "submitter": result[0],
                        "total_submissions": result[1],
                        "approved_submissions": result[2],
                        "rejected_submissions": result[3],
                        "pending_submissions": result[1] - result[2] - result[3],
                        "approval_rate": result[2] / result[1] if result[1] > 0 else 0,
                        "reputation_score": result[4],
                        "is_trusted": bool(result[5]),
                        "is_blocked": bool(result[6]),
                        "first_submission": result[7],
                        "last_submission": result[8]
                    }
                    
                else:
                    # Overall statistics
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_submitters,
                            SUM(total_submissions) as total_submissions,
                            SUM(approved_submissions) as approved_submissions,
                            SUM(rejected_submissions) as rejected_submissions,
                            AVG(reputation_score) as avg_reputation,
                            SUM(CASE WHEN is_trusted = 1 THEN 1 ELSE 0 END) as trusted_submitters,
                            SUM(CASE WHEN is_blocked = 1 THEN 1 ELSE 0 END) as blocked_submitters
                        FROM submitters
                    """)
                    
                    result = cursor.fetchone()
                    
                    # Dataset statistics
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_datasets,
                            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                            SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved,
                            SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected,
                            AVG(quality_score) as avg_quality
                        FROM datasets
                    """)
                    
                    dataset_stats = cursor.fetchone()
                    
                    return {
                        "submitters": {
                            "total": result[0] or 0,
                            "trusted": result[5] or 0,
                            "blocked": result[6] or 0,
                            "average_reputation": result[4] or 0.0
                        },
                        "datasets": {
                            "total": dataset_stats[0] or 0,
                            "pending": dataset_stats[1] or 0,
                            "approved": dataset_stats[2] or 0,
                            "rejected": dataset_stats[3] or 0,
                            "average_quality": dataset_stats[4] or 0.0
                        },
                        "submissions": {
                            "total": result[1] or 0,
                            "approved": result[2] or 0,
                            "rejected": result[3] or 0,
                            "approval_rate": (result[2] or 0) / (result[1] or 1)
                        }
                    }
                    
        except Exception as e:
            self.logger.error(f"Error getting submitter statistics: {e}")
            return {"error": str(e)}

    def manage_submitter(self, submitter: str, action: str, reason: str = None) -> bool:
        """
        Manage submitter status (trust, block, unblock)
        
        Args:
            submitter: Submitter name
            action: Action to take (trust, untrust, block, unblock)
            reason: Reason for action
            
        Returns:
            Success status
        """
        try:
            valid_actions = ["trust", "untrust", "block", "unblock"]
            if action not in valid_actions:
                raise ValueError(f"Invalid action. Must be one of: {valid_actions}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if submitter exists
                cursor.execute("SELECT id FROM submitters WHERE submitter_name = ?", (submitter,))
                if not cursor.fetchone():
                    raise ValueError(f"Submitter {submitter} not found")
                
                # Apply action
                if action == "trust":
                    cursor.execute("""
                        UPDATE submitters 
                        SET is_trusted = TRUE, is_blocked = FALSE,
                            notes = COALESCE(notes || '; ', '') || ?
                        WHERE submitter_name = ?
                    """, (f"Trusted: {reason or 'No reason provided'}", submitter))
                    
                elif action == "untrust":
                    cursor.execute("""
                        UPDATE submitters 
                        SET is_trusted = FALSE,
                            notes = COALESCE(notes || '; ', '') || ?
                        WHERE submitter_name = ?
                    """, (f"Untrusted: {reason or 'No reason provided'}", submitter))
                    
                elif action == "block":
                    cursor.execute("""
                        UPDATE submitters 
                        SET is_blocked = TRUE, is_trusted = FALSE,
                            notes = COALESCE(notes || '; ', '') || ?
                        WHERE submitter_name = ?
                    """, (f"Blocked: {reason or 'No reason provided'}", submitter))
                    
                elif action == "unblock":
                    cursor.execute("""
                        UPDATE submitters 
                        SET is_blocked = FALSE,
                            notes = COALESCE(notes || '; ', '') || ?
                        WHERE submitter_name = ?
                    """, (f"Unblocked: {reason or 'No reason provided'}", submitter))
                
                conn.commit()
                
                self.logger.info(f"Submitter {submitter} {action}ed: {reason}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error managing submitter {submitter}: {e}")
            return False

    def cleanup_old_data(self, days_old: int = 90) -> Dict[str, int]:
        """
        Clean up old rejected datasets and statistics
        
        Args:
            days_old: Age threshold in days
            
        Returns:
            Cleanup statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count datasets to be deleted
                cursor.execute("""
                    SELECT COUNT(*) FROM datasets 
                    WHERE status = 'rejected' AND created_at < ?
                """, (cutoff_date,))
                datasets_to_delete = cursor.fetchone()[0]
                
                # Delete old rejected datasets
                cursor.execute("""
                    DELETE FROM datasets 
                    WHERE status = 'rejected' AND created_at < ?
                """, (cutoff_date,))
                
                # Clean up orphaned feedback
                cursor.execute("""
                    DELETE FROM feedback 
                    WHERE dataset_id NOT IN (SELECT id FROM datasets)
                """)
                feedback_deleted = cursor.rowcount
                
                # Clean up orphaned quality metrics
                cursor.execute("""
                    DELETE FROM quality_metrics 
                    WHERE dataset_id NOT IN (SELECT id FROM datasets)
                """)
                metrics_deleted = cursor.rowcount
                
                conn.commit()
                
                cleanup_stats = {
                    "datasets_deleted": datasets_to_delete,
                    "feedback_deleted": feedback_deleted,
                    "metrics_deleted": metrics_deleted,
                    "cutoff_date": cutoff_date.isoformat()
                }
                
                self.logger.info(f"Cleanup completed: {cleanup_stats}")
                return cleanup_stats
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {"error": str(e)}

    def export_datasets(self, status: str = "approved", format: str = "json") -> str:
        """
        Export datasets in specified format
        
        Args:
            status: Dataset status to export
            format: Export format (json, csv)
            
        Returns:
            Exported data as string
        """
        try:
            datasets = []
            
            if status == "approved":
                datasets = self.get_approved_datasets(limit=10000)
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT id, data, submitter, category, task_type, quality_score
                        FROM datasets WHERE status = ?
                    """, (status,))
                    
                    for row in cursor.fetchall():
                        try:
                            datasets.append({
                                "id": row[0],
                                "data": json.loads(row[1]),
                                "submitter": row[2],
                                "category": row[3],
                                "task_type": row[4],
                                "quality_score": row[5]
                            })
                        except json.JSONDecodeError:
                            continue
            
            if format == "json":
                return json.dumps(datasets, indent=2)
            elif format == "csv":
                # Flatten datasets for CSV export
                import csv
                import io
                
                output = io.StringIO()
                if datasets:
                    fieldnames = ["id", "submitter", "category", "task_type", "quality_score", "data"]
                    writer = csv.DictWriter(output, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for dataset in datasets:
                        row = {
                            "id": dataset["id"],
                            "submitter": dataset["submitter"],
                            "category": dataset["category"],
                            "task_type": dataset["task_type"],
                            "quality_score": dataset["quality_score"],
                            "data": json.dumps(dataset["data"])
                        }
                        writer.writerow(row)
                
                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting datasets: {e}")
            return f"Export error: {str(e)}"

    def get_system_status(self) -> Dict[str, Any]:
        """Get crowdsourcing system status"""
        try:
            stats = self.get_submitter_statistics()
            
            # Add system health metrics
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM datasets 
                    WHERE created_at >= datetime('now', '-7 days')
                """)
                recent_submissions = cursor.fetchone()[0]
                
                # Average review time
                cursor.execute("""
                    SELECT AVG(
                        (julianday(reviewed_at) - julianday(created_at)) * 24
                    ) as avg_review_hours
                    FROM datasets 
                    WHERE reviewed_at IS NOT NULL
                    AND created_at >= datetime('now', '-30 days')
                """)
                avg_review_time = cursor.fetchone()[0]
            
            return {
                "enabled": self.enabled,
                "database_path": self.db_path,
                "statistics": stats,
                "recent_activity": {
                    "submissions_last_7_days": recent_submissions,
                    "average_review_time_hours": avg_review_time or 0
                },
                "system_health": "operational" if self.enabled else "disabled"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e), "system_health": "error"}