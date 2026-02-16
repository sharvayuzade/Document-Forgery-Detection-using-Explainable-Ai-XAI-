"""MongoDB database utilities for predictions storage"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

logger = logging.getLogger(__name__)


class MongoDBManager:
    """Manage MongoDB connections and operations"""
    
    def __init__(self, connection_string: str, db_name: str):
        """
        Initialize MongoDB manager
        
        Args:
            connection_string: MongoDB connection URL
            db_name: Database name
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False
    
    def connect(self) -> bool:
        """
        Connect to MongoDB
        
        Returns:
            True if connection successful
        """
        if not MONGO_AVAILABLE:
            logger.warning("MongoDB not installed. Install pymongo: pip install pymongo")
            return False
        
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.connected = True
            logger.info(f"✓ Connected to MongoDB: {self.db_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"✗ MongoDB connection failed: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def save_prediction(self, file_name: str, prediction: Dict) -> bool:
        """
        Save prediction to database
        
        Args:
            file_name: Name of analyzed file
            prediction: Prediction result dictionary
            
        Returns:
            True if save successful
        """
        if not self.connected:
            return False
        
        try:
            collection = self.db["predictions"]
            document = {
                'file_name': file_name,
                'timestamp': datetime.now(),
                'prediction_score': prediction.get('prediction_score'),
                'status': prediction.get('status'),
                'confidence': prediction.get('confidence'),
                'forgery_score': prediction.get('forgery_score'),
                'suspicious_percentage': prediction.get('suspicious_percentage'),
                'analysis': prediction.get('analysis')
            }
            result = collection.insert_one(document)
            logger.info(f"Saved prediction to MongoDB: {result.inserted_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False
    
    def get_predictions(self, file_name: str = None, limit: int = 10) -> List[Dict]:
        """
        Retrieve predictions from database
        
        Args:
            file_name: Filter by file name (optional)
            limit: Maximum number of results
            
        Returns:
            List of prediction documents
        """
        if not self.connected:
            return []
        
        try:
            collection = self.db["predictions"]
            query = {} if not file_name else {'file_name': file_name}
            results = list(collection.find(query).sort('timestamp', -1).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for doc in results:
                doc['_id'] = str(doc['_id'])
            
            return results
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get statistics from predictions
        
        Returns:
            Dictionary with statistics
        """
        if not self.connected:
            return {}
        
        try:
            collection = self.db["predictions"]
            total = collection.count_documents({})
            authentic = collection.count_documents({'status': 'AUTHENTIC'})
            tampered = collection.count_documents({'status': 'TAMPERED'})
            
            return {
                'total_predictions': total,
                'authentic': authentic,
                'tampered': tampered,
                'tamper_rate': (tampered / total * 100) if total > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def delete_old_predictions(self, days: int = 30) -> int:
        """
        Delete predictions older than N days
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of deleted documents
        """
        if not self.connected:
            return 0
        
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            collection = self.db["predictions"]
            result = collection.delete_many({'timestamp': {'$lt': cutoff_date}})
            logger.info(f"Deleted {result.deleted_count} old predictions")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting old predictions: {e}")
            return 0


# Global MongoDB manager instance
db_manager = None


def init_database(connection_string: str, db_name: str) -> Optional[MongoDBManager]:
    """
    Initialize global database manager
    
    Args:
        connection_string: MongoDB URL
        db_name: Database name
        
    Returns:
        MongoDBManager instance or None
    """
    global db_manager
    db_manager = MongoDBManager(connection_string, db_name)
    if db_manager.connect():
        return db_manager
    return None


def get_db_manager() -> Optional[MongoDBManager]:
    """Get global database manager instance"""
    return db_manager
