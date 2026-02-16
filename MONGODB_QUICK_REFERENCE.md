# MongoDB Integration - Quick Reference

## Quick Start

### 1. Verify MongoDB Connection

```bash
python test_mongodb_integration.py
```

This runs comprehensive tests including:
- MongoDB connection
- Saving predictions
- Querying predictions
- Statistics generation
- API endpoint testing
- Automatic cleanup

### 2. Configuration

Update `src/config.py` if MongoDB is on a different host:

```python
MONGODB_URL = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "forgery_detection"
```

### 3. Start the API

```bash
# Terminal 1: Start FastAPI backend
python -m backend.app

# API will be running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## API Quick Reference

### Analyze Image (Auto-saves to MongoDB)
```bash
curl -X POST http://localhost:8000/api/analyze/image \
  -F "file=@document.jpg"
```

### Get Recent Predictions
```bash
curl http://localhost:8000/api/predictions/recent?limit=10
```

### Get Predictions for Specific File
```bash
curl "http://localhost:8000/api/predictions/file/document.jpg"
```

### Get Statistics
```bash
curl http://localhost:8000/api/statistics
```

### Check API Health
```bash
curl http://localhost:8000/api/health
```

## Python Usage

### Direct Database Access

```python
from src.utils.database import init_database, get_db_manager
from src.config import MONGODB_URL, MONGODB_DB_NAME

# Initialize database
db_manager = init_database(MONGODB_URL, MONGODB_DB_NAME)

# Save prediction
record = {
    'file_name': 'document.jpg',
    'prediction_score': 0.87,
    'status': 'TAMPERED',
    'confidence': 0.87,
    'forgery_score': 0.92,
    'suspicious_percentage': 42.5,
    'analysis': {...}
}
db_manager.save_prediction(record)

# Get predictions
predictions = db_manager.get_predictions(limit=10)

# Get statistics
stats = db_manager.get_statistics()

# Query specific file
file_predictions = db_manager.get_predictions(
    query={'file_name': 'document.jpg'}
)
```

### REST API Client

```python
import requests

BASE_URL = "http://localhost:8000/api"

# Analyze image
with open("document.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze/image",
        files={"file": f}
    )
    result = response.json()

# Get recent predictions
response = requests.get(f"{BASE_URL}/predictions/recent?limit=5")
predictions = response.json()

# Get statistics
response = requests.get(f"{BASE_URL}/statistics")
stats = response.json()
```

## MongoDB CLI Commands

### Check Connection
```bash
mongosh mongodb://localhost:27017/
```

### View Data
```javascript
// Show all databases
show databases

// Use forgery_detection database
use forgery_detection

// Show collections
show collections

// View sample prediction
db.predictions.findOne()

// Count predictions
db.predictions.countDocuments()

// View all predictions
db.predictions.find().pretty()
```

### Query Examples
```javascript
// Tampered documents
db.predictions.find({status: "TAMPERED"})

// Recent predictions (last 24 hours)
db.predictions.find({
  timestamp: {$gte: new Date(new Date() - 24*60*60*1000)}
})

// High confidence predictions
db.predictions.find({confidence: {$gt: 0.9}})

// Specific file
db.predictions.find({file_name: "document.jpg"})
```

### Create Indexes (for better performance)
```javascript
db.predictions.createIndex({file_name: 1})
db.predictions.createIndex({timestamp: -1})
db.predictions.createIndex({status: 1})
```

## Troubleshooting

### MongoDB Not Running
```bash
# Linux
sudo systemctl start mongod

# Windows
net start MongoDB

# macOS
brew services start mongodb-community

# Docker
docker run -d -p 27017:27017 --name mongodb mongo
```

### Test Connection
```bash
# Check if MongoDB is accessible
mongosh mongodb://localhost:27017/

# If fails, check if service is running and port is correct
```

### Reset Test Data
```bash
# Delete all test predictions from MongoDB
mongo forgery_detection --eval "db.predictions.deleteMany({file_name: /test_document/})"
```

### Check Database Size
```javascript
db.stats()
```

### Backup Database
```bash
mongodump --db forgery_detection --out backup/
```

### Restore Database
```bash
mongorestore --db forgery_detection backup/forgery_detection/
```

## Environment Variables

Create `.env` file in project root:

```dotenv
MONGODB_URL=mongodb://localhost:27017/
MONGODB_DB_NAME=forgery_detection
```

Or set in shell:
```bash
export MONGODB_URL=mongodb://localhost:27017/
export MONGODB_DB_NAME=forgery_detection
```

## Performance Tips

1. **Create Indexes** for frequently queried fields
2. **Use Limits** when querying large result sets
3. **Archive Old Data** to keep collection size manageable
4. **Monitor Connection Pool** - info in logs
5. **Use Batch Operations** for bulk saves

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Connection refused | Start MongoDB service |
| Database operation timed out | Check MongoDB load, consider indexes |
| High memory usage | Implement data retention policy |
| Slow queries | Create indexes on query fields |
| Duplicate predictions | Query DB before saving new record |

## File Locations

| Component | Location |
|-----------|----------|
| Config | `src/config.py` |
| Database Manager | `src/utils/database.py` |
| Backend API | `backend/app.py` |
| Test Script | `test_mongodb_integration.py` |
| Documentation | `MONGODB_INTEGRATION.md` |
| Environment Template | `.env.example` |

## Next Steps

1. ✓ MongoDB installed and running
2. ✓ Dependencies installed (`pymongo`, `motorengine`)
3. ✓ Backend API verified with test script
4. ✓ Analyze documents (automatically saved to DB)
5. ✓ Query predictions via API endpoints
6. ✓ Monitor statistics and trends

For detailed information, see `MONGODB_INTEGRATION.md`
