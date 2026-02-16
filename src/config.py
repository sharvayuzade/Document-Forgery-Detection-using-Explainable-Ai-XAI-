"""
Configuration file for Document Forgery Detection using XAI
Optimized for Intel i5-1220P with Intel Iris Xe GPU
"""

import os
from pathlib import Path

# ==================== PROJECT PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "archive" / "CASIA2"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [MODELS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset paths
AUTHENTIC_DIR = DATA_DIR / "Au"
TAMPERED_DIR = DATA_DIR / "Tp"
GROUNDTRUTH_DIR = DATA_DIR / "CASIA 2 Groundtruth"

# ==================== HARDWARE CONFIGURATION ====================
# Optimized for Intel i5-1220P with Iris Xe GPU
DEVICE = "GPU"  # Use GPU when available
USE_MIXED_PRECISION = True  # Reduce memory usage
MAX_WORKERS = 4  # CPU threads for data loading
BATCH_SIZE = 16  # Smaller batch size for limited GPU memory
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

# ==================== MODEL CONFIGURATION ====================
MODEL_NAME = "ResNet50"  # Lightweight and fast
INPUT_SIZE = 224  # Standard input size
NUM_CLASSES = 2  # Binary: Authentic (0) vs Tampered (1)
PRETRAINED = True  # Use ImageNet pretraining

# Training parameters
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9
PATIENCE = 5  # Early stopping patience
MIN_DELTA = 0.0001

# ==================== DATA SPLIT ====================
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==================== PREPROCESSING ====================
# Error Level Analysis (ELA)
ELA_QUALITY = 90  # JPEG compression quality for ELA
ELA_SCALE = 255  # Scale factor for visualization

# Normalization parameters (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data augmentation
AUGMENTATION = {
    'rotation_range': 10,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# ==================== PDF PROCESSING ====================
PDF_DPI = 150  # DPI for PDF to image conversion (optimized for speed)
PDF_MAX_PAGES = 50  # Limit pages to process
PDF_FORMAT = "png"

# ==================== GRAD-CAM CONFIGURATION ====================
GRADCAM_LAYER = "conv5_block3_3_bn"  # Last convolutional layer for ResNet50
HEATMAP_ALPHA = 0.4  # Transparency for heatmap overlay
HEATMAP_COLORMAP = "jet"  # Colormap for visualization

# ==================== OUTPUT SETTINGS ====================
SAVE_HEATMAPS = True
SAVE_PREDICTIONS = True
CONFIDENCE_THRESHOLD = 0.5  # Decision threshold

# ==================== LOGGING ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ==================== PATHS FOR MODELS ====================
BEST_MODEL_PATH = MODELS_DIR / "best_model.h5"
LATEST_MODEL_PATH = MODELS_DIR / "latest_model.h5"
MODEL_WEIGHTS_PATH = MODELS_DIR / "model_weights.h5"

# ==================== API CONFIGURATION ====================
API_HOST = "127.0.0.1"
API_PORT = 8000
API_RELOAD = True
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

# ==================== DATABASE CONFIGURATION ====================
MONGODB_URL = "mongodb://localhost:27017/"
MONGODB_DB_NAME = "forgery_detection"
ENABLE_DATABASE = True

# Collections
PREDICTIONS_COLLECTION = "predictions"
HEATMAPS_COLLECTION = "heatmaps"
USERS_COLLECTION = "users"

# ==================== FRONTEND CONFIGURATION ====================
STREAMLIT_PORT = 8501

print(f"✓ Configuration loaded successfully")
print(f"✓ Project root: {PROJECT_ROOT}")
print(f"✓ Data directory: {DATA_DIR}")
print(f"✓ Batch size: {BATCH_SIZE} (optimized for Intel i5-1220P)")
print(f"✓ Device: {DEVICE}")
print(f"✓ Mixed precision training: {USE_MIXED_PRECISION}")
