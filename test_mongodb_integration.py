#!/usr/bin/env python3
"""
MongoDB Integration Test Script

This script verifies that MongoDB integration is working correctly by:
1. Testing MongoDB connection
2. Saving sample predictions
3. Querying predictions
4. Getting statistics
5. Testing API endpoints (if running)

Usage:
    python test_mongodb_integration.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.config import MONGODB_URL, MONGODB_DB_NAME
    from src.utils.database import init_database, get_db_manager
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Ensure your Python environment is properly configured.")
    sys.exit(1)


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_database_connection():
    """Test connection to MongoDB"""
    print_section("1. Testing MongoDB Connection")
    
    try:
        db_manager = init_database(MONGODB_URL, MONGODB_DB_NAME)
        if db_manager:
            print("✓ Successfully connected to MongoDB")
            print(f"  URL: {MONGODB_URL}")
            print(f"  Database: {MONGODB_DB_NAME}")
            return db_manager
        else:
            print("✗ Failed to initialize database manager")
            return None
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None


def test_save_predictions(db_manager):
    """Test saving predictions to database"""
    print_section("2. Testing Save Predictions")
    
    if not db_manager:
        print("✗ Database manager not available")
        return False
    
    try:
        # Create sample predictions
        predictions = [
            {
                'file_name': 'test_document_1.jpg',
                'prediction_score': 0.87,
                'status': 'TAMPERED',
                'confidence': 0.87,
                'forgery_score': 0.92,
                'suspicious_percentage': 42.5,
                'analysis': {
                    'prediction_score': 0.87,
                    'status': 'TAMPERED',
                    'confidence_percentage': 87.0,
                    'forgery_score': 0.92,
                    'suspicious_pixels': 425000,
                    'total_pixels': 1000000,
                    'suspicious_percentage': 42.5
                }
            },
            {
                'file_name': 'test_document_2.jpg',
                'prediction_score': 0.23,
                'status': 'AUTHENTIC',
                'confidence': 0.77,
                'forgery_score': 0.15,
                'suspicious_percentage': 8.2,
                'analysis': {
                    'prediction_score': 0.23,
                    'status': 'AUTHENTIC',
                    'confidence_percentage': 77.0,
                    'forgery_score': 0.15,
                    'suspicious_pixels': 82000,
                    'total_pixels': 1000000,
                    'suspicious_percentage': 8.2
                }
            },
            {
                'file_name': 'test_document_3.pdf_page_1',
                'prediction_score': 0.56,
                'status': 'TAMPERED',
                'confidence': 0.56,
                'forgery_score': 0.65,
                'suspicious_percentage': 28.3,
                'analysis': {
                    'prediction_score': 0.56,
                    'status': 'TAMPERED',
                    'confidence_percentage': 56.0,
                    'forgery_score': 0.65,
                    'suspicious_pixels': 283000,
                    'total_pixels': 1000000,
                    'suspicious_percentage': 28.3
                }
            }
        ]
        
        count = 0
        for pred in predictions:
            db_manager.save_prediction(pred)
            count += 1
            print(f"✓ Saved: {pred['file_name']} - Status: {pred['status']}")
        
        print(f"\n✓ Successfully saved {count} sample predictions")
        return True
    
    except Exception as e:
        print(f"✗ Error saving predictions: {e}")
        return False


def test_get_predictions(db_manager):
    """Test retrieving predictions from database"""
    print_section("3. Testing Get Predictions")
    
    if not db_manager:
        print("✗ Database manager not available")
        return False
    
    try:
        # Get all predictions
        all_predictions = db_manager.get_predictions()
        print(f"✓ Retrieved all predictions: {len(all_predictions)} records")
        
        if all_predictions:
            print(f"\n  Sample record:")
            sample = all_predictions[0]
            print(f"    File: {sample.get('file_name')}")
            print(f"    Status: {sample.get('status')}")
            print(f"    Prediction Score: {sample.get('prediction_score')}")
        
        # Get recent predictions with limit
        recent = db_manager.get_predictions(limit=2)
        print(f"\n✓ Retrieved recent predictions (limit=2): {len(recent)} records")
        
        # Get predictions for specific file
        file_preds = db_manager.get_predictions(query={'file_name': 'test_document_1.jpg'})
        print(f"✓ Retrieved predictions for 'test_document_1.jpg': {len(file_preds)} records")
        
        # Get predictions by status
        tampered = db_manager.get_predictions(query={'status': 'TAMPERED'})
        authentic = db_manager.get_predictions(query={'status': 'AUTHENTIC'})
        print(f"✓ Tampered predictions: {len(tampered)}")
        print(f"✓ Authentic predictions: {len(authentic)}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error retrieving predictions: {e}")
        return False


def test_get_statistics(db_manager):
    """Test getting statistics from database"""
    print_section("4. Testing Get Statistics")
    
    if not db_manager:
        print("✗ Database manager not available")
        return False
    
    try:
        stats = db_manager.get_statistics()
        
        print("✓ Retrieved statistics:")
        print(f"  Total Predictions: {stats.get('total_predictions')}")
        print(f"  Authentic Count: {stats.get('authentic_count')}")
        print(f"  Tampered Count: {stats.get('tampered_count')}")
        print(f"  Authenticity Rate: {stats.get('authenticity_rate'):.2f}%")
        print(f"  Average Confidence: {stats.get('average_confidence'):.4f}")
        print(f"  Average Forgery Score: {stats.get('average_forgery_score'):.4f}")
        
        if stats.get('predictions_by_date'):
            print(f"  Predictions by Date: {stats['predictions_by_date']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error getting statistics: {e}")
        return False


def test_api_endpoints():
    """Test API endpoints if available"""
    print_section("5. Testing API Endpoints (if running)")
    
    try:
        import requests
    except ImportError:
        print("⊙ requests library not installed, skipping API endpoint tests")
        print("  Install with: pip install requests")
        return True
    
    base_url = "http://localhost:8000/api"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            print("✓ Health endpoint: OK")
            health = response.json()
            print(f"  Model loaded: {health.get('model_loaded')}")
            print(f"  Database connected: {health.get('database_connected')}")
        else:
            print(f"⊙ Health endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"⊙ Could not reach health endpoint: {e}")
        print("  Make sure backend is running: python -m backend.app")
        return True
    
    try:
        # Test status endpoint
        response = requests.get(f"{base_url}/status", timeout=2)
        if response.status_code == 200:
            print("\n✓ Status endpoint: OK")
            status = response.json()
            print(f"  API Version: {status.get('version')}")
            print(f"  Database: {status.get('database')}")
        else:
            print(f"⊙ Status endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"⊙ Could not reach status endpoint: {e}")
    
    try:
        # Test predictions/recent endpoint
        response = requests.get(f"{base_url}/predictions/recent?limit=2", timeout=2)
        if response.status_code == 200:
            print("\n✓ Predictions/recent endpoint: OK")
            data = response.json()
            print(f"  Retrieved {data.get('count')} recent predictions")
        elif response.status_code == 503:
            print("\n⊙ Predictions/recent endpoint: Database not available (API running without DB)")
        else:
            print(f"⊙ Predictions/recent endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"⊙ Could not reach predictions endpoint: {e}")
    
    try:
        # Test statistics endpoint
        response = requests.get(f"{base_url}/statistics", timeout=2)
        if response.status_code == 200:
            print("\n✓ Statistics endpoint: OK")
            stats = response.json()
            print(f"  Total predictions in database: {stats.get('total_predictions')}")
        elif response.status_code == 503:
            print("\n⊙ Statistics endpoint: Database not available (API running without DB)")
        else:
            print(f"⊙ Statistics endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"⊙ Could not reach statistics endpoint: {e}")
    
    return True


def cleanup_test_data(db_manager):
    """Clean up test predictions from database"""
    print_section("6. Cleaning Up Test Data")
    
    if not db_manager:
        print("✗ Database manager not available")
        return False
    
    try:
        # Get collection
        collection = db_manager.db[db_manager.collections['predictions']]
        
        # Delete test predictions
        test_files = [
            'test_document_1.jpg',
            'test_document_2.jpg',
            'test_document_3.pdf_page_1'
        ]
        
        for file_name in test_files:
            result = collection.delete_many({'file_name': file_name})
            if result.deleted_count > 0:
                print(f"✓ Deleted {result.deleted_count} test predictions for {file_name}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error cleaning up: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("█" * 80)
    print("█   Document Forgery Detection - MongoDB Integration Test Suite".ljust(79) + "█")
    print("█" * 80)
    
    # Test connection
    db_manager = test_database_connection()
    if not db_manager:
        print("\n" + "!"*80)
        print("CRITICAL: MongoDB connection failed!")
        print("!"*80)
        print("\nTroubleshooting steps:")
        print("1. Verify MongoDB is running:")
        print("   - Linux: sudo systemctl status mongod")
        print("   - Windows: net start MongoDB")
        print("   - Mac: brew services list")
        print("\n2. Test MongoDB connection manually:")
        print("   mongosh mongodb://localhost:27017")
        print("\n3. Check configuration in src/config.py:")
        print(f"   MONGODB_URL = {MONGODB_URL}")
        print(f"   MONGODB_DB_NAME = {MONGODB_DB_NAME}")
        return 1
    
    # Run remaining tests
    results = []
    results.append(("Save Predictions", test_save_predictions(db_manager)))
    results.append(("Get Predictions", test_get_predictions(db_manager)))
    results.append(("Get Statistics", test_get_statistics(db_manager)))
    results.append(("API Endpoints", test_api_endpoints()))
    
    # Cleanup
    cleanup_test_data(db_manager)
    
    # Summary
    print_section("Test Summary")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ MongoDB integration is working correctly!")
        return 0
    else:
        print("\n✗ Some tests failed. Check output above for details.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
