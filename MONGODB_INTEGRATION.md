# MongoDB Integration Guide

## Overview

The Document Forgery Detection API now includes MongoDB integration for persistent storage of predictions and analysis results. This enables:

- **Historical Tracking**: Store all predictions with timestamps for future reference
- **Query Capabilities**: Retrieve predictions by file name or date range
- **Statistical Analysis**: Generate insights from accumulated analysis data
- **Audit Trail**: Maintain compliance by tracking all document analyses

## Configuration

### Prerequisites

1. MongoDB server running and accessible at configured URL (default: `mongodb://localhost:27017/`)
2. Python packages installed: `pymongo>=4.5.0`, `motorengine>=0.9.0`

### Environment Setup

```bash
# Install MongoDB integration packages
pip install pymongo==4.5.0 motorengine==0.9.0
```

### Configuration in src/config.py

The following variables control MongoDB behavior:

```python
# MongoDB Configuration
MONGODB_URL = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "forgery_detection"
MONGODB_COLLECTIONS = {
    'predictions': 'predictions',
    'heatmaps': 'heatmaps',
    'users': 'users'
}
```

**Note**: Change `MONGODB_URL` if your MongoDB instance is running on a different host or port.

## Database Schema

### Predictions Collection

Each prediction document contains:

```json
{
  "_id": ObjectId,
  "file_name": "string",
  "timestamp": "datetime",
  "prediction_score": "float",
  "status": "AUTHENTIC | TAMPERED",
  "confidence": "float",
  "forgery_score": "float",
  "suspicious_percentage": "float",
  "analysis": {
    "prediction_score": "float",
    "status": "AUTHENTIC | TAMPERED",
    "confidence_percentage": "float",
    "forgery_score": "float",
    "suspicious_pixels": "integer",
    "total_pixels": "integer",
    "suspicious_percentage": "float"
  }
}
```

### Heatmaps Collection

Stores reference data for heatmap visualizations and analysis images.

### Users Collection

Can be extended to track which user performed each analysis.

## API Endpoints

### New Database Query Endpoints

#### 1. Get Recent Predictions

```
GET /api/predictions/recent?limit=10
```

**Parameters:**
- `limit` (int, optional): Number of recent predictions (default: 10)

**Response:**
```json
{
  "count": 5,
  "predictions": [
    {
      "_id": "...",
      "file_name": "document.jpg",
      "timestamp": "2024-01-15T10:30:45.123Z",
      "prediction_score": 0.87,
      "status": "TAMPERED",
      "confidence": 0.87,
      "forgery_score": 0.92,
      "suspicious_percentage": 42.5,
      "analysis": {...}
    },
    ...
  ]
}
```

#### 2. Get Predictions for Specific File

```
GET /api/predictions/file/{filename}
```

**Parameters:**
- `filename` (string): Name of the file to query

**Response:**
```json
{
  "file_name": "document.jpg",
  "count": 3,
  "predictions": [...]
}
```

**Example:**
```
GET /api/predictions/file/suspicious_document.pdf
```

#### 3. Get Statistics

```
GET /api/statistics
```

**Response:**
```json
{
  "total_predictions": 245,
  "authentic_count": 156,
  "tampered_count": 89,
  "authenticity_rate": 63.67,
  "average_confidence": 0.845,
  "average_forgery_score": 0.523,
  "predictions_by_date": {
    "2024-01-15": 12,
    "2024-01-14": 8,
    ...
  }
}
```

### Existing Endpoints Now with Database Persistence

#### Single Image Analysis

```
POST /api/analyze/image
```

**Behavior**: After analysis produces a prediction, it is automatically saved to MongoDB with:
- File name extracted from upload
- Timestamp set to current time
- All analysis results stored in database

#### PDF Analysis

```
POST /api/analyze/pdf
```

**Behavior**: Each page is analyzed and each page's prediction is saved to MongoDB as a separate record with:
- File name: `{original_filename}_page_{page_number}`
- Individual page analysis data

#### Batch Analysis

```
POST /api/batch/analyze
```

**Behavior**: All files in batch are analyzed and saved to MongoDB with:
- File name for each file
- Page numbers for PDF pages
- Complete analysis results

## Database Manager Class

Location: `src/utils/database.py`

### Key Methods

#### `connect()`
Establishes connection to MongoDB server.

```python
db_manager.connect()
```

#### `save_prediction(record: dict)`
Saves a prediction record to MongoDB.

```python
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
```

#### `get_predictions(query: dict = None, limit: int = None)`
Retrieves predictions from database with optional filtering.

```python
# Get all predictions
all_predictions = db_manager.get_predictions()

# Get specific file predictions
file_predictions = db_manager.get_predictions(
    query={'file_name': 'document.jpg'}
)

# Get recent predictions (limit)
recent = db_manager.get_predictions(limit=10)
```

#### `get_statistics()`
Returns aggregated statistics from predictions collection.

```python
stats = db_manager.get_statistics()
# Returns: {
#   'total_predictions': 245,
#   'authentic_count': 156,
#   'tampered_count': 89,
#   'authenticity_rate': 63.67,
#   'average_confidence': 0.845,
#   'average_forgery_score': 0.523,
#   'predictions_by_date': {...}
# }
```

#### `close()`
Closes MongoDB connection.

```python
db_manager.close()
```

## API Startup Integration

The MongoDB connection is automatically initialized when the API starts:

```python
# In backend/app.py startup_event()
db_manager = init_database(MONGODB_URL, MONGODB_DB_NAME)
```

### Graceful Degradation

If MongoDB is unavailable:
- API starts successfully but without database functionality
- Database query endpoints return 503 Service Unavailable
- Predictions still work via FastAPI endpoints but are not persisted
- Error messages logged for debugging

## Testing MongoDB Integration

### Manual Test with curl

```bash
# 1. Analyze an image
curl -X POST http://localhost:8000/api/analyze/image \
  -F "file=@test_image.jpg"

# 2. Check if it was saved
curl http://localhost:8000/api/predictions/recent?limit=1

# 3. Get statistics
curl http://localhost:8000/api/statistics

# 4. Query specific file
curl http://localhost:8000/api/predictions/file/test_image.jpg
```

### Python Client Test

```python
import requests
import json

BASE_URL = "http://localhost:8000/api"

# 1. Analyze image
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/analyze/image",
        files={"file": f}
    )
    result = response.json()
    print(f"Analysis Result: {json.dumps(result, indent=2)}")

# 2. Get recent predictions
response = requests.get(f"{BASE_URL}/predictions/recent?limit=5")
predictions = response.json()
print(f"Recent Predictions: {json.dumps(predictions, indent=2)}")

# 3. Get statistics
response = requests.get(f"{BASE_URL}/statistics")
stats = response.json()
print(f"Statistics: {json.dumps(stats, indent=2)}")

# 4. Query specific file
response = requests.get(f"{BASE_URL}/predictions/file/test_image.jpg")
file_preds = response.json()
print(f"File Predictions: {json.dumps(file_preds, indent=2)}")
```

## MongoDB Administration

### View Predictions in MongoDB

```bash
# Connect to MongoDB
mongosh

# Select database
use forgery_detection

# View all predictions
db.predictions.find().pretty()

# View predictions by status
db.predictions.find({status: "TAMPERED"}).pretty()

# Count predictions
db.predictions.countDocuments()

# Get average prediction score
db.predictions.aggregate([
  {
    $group: {
      _id: null,
      avg_score: { $avg: "$prediction_score" }
    }
  }
])

# Get predictions by date range
db.predictions.find({
  timestamp: {
    $gte: new Date("2024-01-01"),
    $lt: new Date("2024-02-01")
  }
})
```

## Troubleshooting

### MongoDB Connection Failed

**Error**: `Connection to MongoDB failed: [error message]`

**Solution**:
1. Verify MongoDB is running: `sudo systemctl status mongod` (Linux)
2. Check connection string in `src/config.py`
3. Verify MongoDB is accessible: `mongosh mongodb://localhost:27017/`

### Database Query Returns Empty

**Possible Causes**:
- No predictions have been analyzed yet
- Predictions not being saved (MongoDB unavailable during analysis)
- Incorrect file name in query

**Solution**:
1. Analyze some images first
2. Check API logs for database errors
3. Verify file names in query match exactly

### Large Database Performance Issues

**Solution**:
- Create indexes on frequently queried fields:
  ```javascript
  db.predictions.createIndex({file_name: 1})
  db.predictions.createIndex({timestamp: -1})
  db.predictions.createIndex({status: 1})
  ```

- Use pagination for large result sets

## Performance Considerations

- **Batch Indexes**: Create indexes for improved query performance
- **Data Retention**: Implement cleanup policies for old predictions
- **Connection Pooling**: MongoDB driver uses connection pooling automatically
- **Async Operations**: Database operations are non-blocking in FastAPI

## Security

### Protect MongoDB Access

```python
# In production, use authentication
MONGODB_URL = "mongodb://username:password@host:port/"
```

### Enable MongoDB Authentication

```javascript
// In MongoDB
use admin
db.createUser({
  user: "forgery_api",
  pwd: "strong_password",
  roles: [{role: "readWrite", db: "forgery_detection"}]
})
```

## Future Enhancements

- User authentication and per-user prediction tracking
- Advanced analytics and reporting dashboards
- Prediction export features (CSV, PDF reports)
- Integration with notification systems
- Machine learning on prediction patterns for model improvement

## References

- [MongoDB Python Driver](https://pymongo.readthedocs.io/)
- [FastAPI Database Integration](https://fastapi.tiangolo.com/advanced/sql-databases/)
- [MongoDB Query Documentation](https://docs.mongodb.com/manual/reference/operator/query/)
