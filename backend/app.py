"""
FastAPI backend for document forgery detection with Grad-CAM explanations
Serves the model predictions and XAI visualizations
"""

import os
import sys
import tempfile
import base64
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import *
from data.data_loader import PDFProcessor
from utils.preprocessing import ImagePreprocessor, ErrorLevelAnalysis
from xai.gradcam import GradCAM
from utils.database import init_database, get_db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Forgery Detection API",
    description="XAI-powered API for detecting forged documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
gradcam = None
preprocessor = None
pdf_processor = None
ela = None
db_manager = None


@app.on_event("startup")
async def startup_event():
    """Load model and initialize components on startup"""
    global model, gradcam, preprocessor, pdf_processor, ela, db_manager
    
    logger.info("="*80)
    logger.info("Initializing Document Forgery Detection API")
    logger.info("="*80)
    
    try:
        # Initialize database connection
        logger.info(f"Connecting to MongoDB at {MONGODB_URL}")
        db_manager = init_database(MONGODB_URL, MONGODB_DB_NAME)
        logger.info("✓ Database connection initialized")
        
        # Load model
        if BEST_MODEL_PATH.exists():
            logger.info(f"Loading model from {BEST_MODEL_PATH}")
            model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        else:
            logger.error(f"Model not found at {BEST_MODEL_PATH}")
            raise FileNotFoundError(f"Model not found at {BEST_MODEL_PATH}")
        
        # Find appropriate layer for Grad-CAM
        # For ResNet50: 'conv5_block3_3_bn'
        # For MobileNetV2: 'out_relu'
        # Try to find automatically
        layer_name = None
        for layer in model.layers:
            if 'conv' in layer.name and 'bn' in layer.name:
                layer_name = layer.name
        
        if layer_name is None:
            # Fallback to last layer before output
            layer_name = model.layers[-3].name
        
        logger.info(f"Using layer {layer_name} for Grad-CAM")
        gradcam = GradCAM(model, layer_name)
        
        # Initialize other components
        preprocessor = ImagePreprocessor(target_size=(INPUT_SIZE, INPUT_SIZE))
        pdf_processor = PDFProcessor(dpi=PDF_DPI, max_pages=PDF_MAX_PAGES)
        ela = ErrorLevelAnalysis(quality=ELA_QUALITY, scale=ELA_SCALE)
        
        logger.info("✓ Model and components initialized successfully")
        logger.info("="*80 + "\n")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64


def process_single_image(image: np.ndarray, image_path: str = None) -> dict:
    """
    Process single image and generate prediction with Grad-CAM
    
    Args:
        image: Image as numpy array (BGR)
        image_path: Optional path to image for reference
        
    Returns:
        Dictionary with prediction, confidence, and visualizations
    """
    try:
        # Preprocess image
        preprocessed = preprocessor.preprocess_image(image, apply_ela=False, normalize=True)
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        
        # Get prediction
        prediction = model.predict(preprocessed, verbose=0)[0][0]
        
        # Compute Grad-CAM
        heatmap = gradcam.compute_gradcam(preprocessed)[0]
        
        # Create overlay
        overlay = gradcam.overlay_heatmap(image, heatmap, alpha=HEATMAP_ALPHA)
        
        # Generate report
        report = gradcam.generate_report(image, prediction, heatmap, CONFIDENCE_THRESHOLD)
        
        # Prepare response
        result = {
            'prediction_score': float(prediction),
            'is_tampered': bool(prediction > CONFIDENCE_THRESHOLD),
            'confidence': float(report['confidence']),
            'forgery_score': float(report['forgery_score']),
            'suspicious_percentage': float(report['suspicious_percentage']),
            'overlay_image': image_to_base64(overlay),
            'heatmap': image_to_base64((heatmap * 255).astype(np.uint8)),
            'suspicious_mask': image_to_base64(report['suspicious_mask']),
            'analysis': {
                'prediction_score': float(prediction),
                'status': 'TAMPERED' if prediction > CONFIDENCE_THRESHOLD else 'AUTHENTIC',
                'confidence_percentage': float(report['confidence'] * 100),
                'forgery_score': float(report['forgery_score']),
                'suspicious_pixels': report['suspicious_pixels'],
                'total_pixels': report['total_pixels'],
                'suspicious_percentage': float(report['suspicious_percentage'])
            }
        }
        
        # Save to database if available
        if db_manager:
            try:
                db_record = {
                    'file_name': image_path or 'unknown',
                    'prediction_score': float(prediction),
                    'status': 'TAMPERED' if prediction > CONFIDENCE_THRESHOLD else 'AUTHENTIC',
                    'confidence': float(report['confidence']),
                    'forgery_score': float(report['forgery_score']),
                    'suspicious_percentage': float(report['suspicious_percentage']),
                    'analysis': result['analysis']
                }
                db_manager.save_prediction(db_record)
            except Exception as db_error:
                logger.warning(f"Could not save prediction to database: {db_error}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze single image file
    
    Args:
        file: Image file (JPG, PNG, etc.)
        
    Returns:
        JSON with prediction and visualizations
    """
    try:
        # Validate file type
        if file.content_type not in ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff']:
            raise HTTPException(status_code=400, detail="Invalid image format. Supported: JPG, PNG, BMP, TIFF")
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Load image
            image = cv2.imread(tmp_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Could not read image file")
            
            # Process
            result = process_single_image(image, file.filename)
            
            return JSONResponse(status_code=200, content=result)
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in API: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze PDF file (all pages)
    
    Args:
        file: PDF file
        
    Returns:
        JSON with analysis for each page
    """
    try:
        # Validate file type
        if file.content_type != 'application/pdf' and not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Invalid file. Only PDF files are supported")
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Convert PDF to images
            images = pdf_processor.pdf_to_images(tmp_path)
            
            if len(images) == 0:
                raise HTTPException(status_code=400, detail="Could not extract images from PDF")
            
            # Process each page
            pages_analysis = []
            for page_num, image in enumerate(images):
                try:
                    result = process_single_image(image, f"page_{page_num+1}")
                    result['page_number'] = page_num + 1
                    pages_analysis.append(result)
                except Exception as e:
                    logger.warning(f"Error processing page {page_num+1}: {e}")
                    pages_analysis.append({
                        'page_number': page_num + 1,
                        'error': str(e)
                    })
            
            # Summary
            tampered_pages = sum(1 for p in pages_analysis if p.get('is_tampered', False))
            total_pages = len(pages_analysis)
            
            response = {
                'total_pages': total_pages,
                'tampered_pages': tampered_pages,
                'integrity_status': 'SUSPICIOUS' if tampered_pages > 0 else 'AUTHENTIC',
                'pages': pages_analysis,
                'summary': {
                    'total_pages': total_pages,
                    'authentic_pages': total_pages - tampered_pages,
                    'tampered_pages': tampered_pages,
                    'integrity_percentage': float(100 * (1 - tampered_pages / total_pages))
                }
            }
            
            return JSONResponse(status_code=200, content=response)
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in PDF analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch/analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """
    Batch analyze multiple files
    
    Args:
        files: List of image or PDF files
        
    Returns:
        JSON with analysis for all files
    """
    try:
        results = []
        
        for file in files:
            try:
                if file.content_type == 'application/pdf' or file.filename.endswith('.pdf'):
                    # Process as PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        contents = await file.read()
                        tmp.write(contents)
                        tmp_path = tmp.name
                    
                    try:
                        images = pdf_processor.pdf_to_images(tmp_path)
                        for i, img in enumerate(images):
                            analysis = process_single_image(img, f"{file.filename}_page_{i+1}")
                            analysis['file'] = file.filename
                            analysis['page'] = i + 1
                            results.append(analysis)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                else:
                    # Process as image
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        contents = await file.read()
                        tmp.write(contents)
                        tmp_path = tmp.name
                    
                    try:
                        image = cv2.imread(tmp_path)
                        analysis = process_single_image(image, file.filename)
                        analysis['file'] = file.filename
                        results.append(analysis)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
            
            except Exception as e:
                logger.warning(f"Error processing {file.filename}: {e}")
                results.append({
                    'file': file.filename,
                    'error': str(e)
                })
        
        return JSONResponse(status_code=200, content={'results': results})
    
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/recent")
async def get_recent_predictions(limit: int = 10):
    """
    Get recent predictions from database
    
    Args:
        limit: Number of recent predictions to retrieve (default: 10)
        
    Returns:
        JSON with list of recent predictions
    """
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        predictions = db_manager.get_predictions(limit=limit)
        return JSONResponse(
            status_code=200,
            content={
                'count': len(predictions),
                'predictions': predictions
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/file/{filename}")
async def get_file_predictions(filename: str):
    """
    Get all predictions for a specific file
    
    Args:
        filename: Name of the file to query
        
    Returns:
        JSON with all predictions for the file
    """
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        predictions = db_manager.get_predictions(query={'file_name': filename})
        return JSONResponse(
            status_code=200,
            content={
                'file_name': filename,
                'count': len(predictions),
                'predictions': predictions
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving file predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics():
    """
    Get aggregated statistics from predictions
    
    Returns:
        JSON with database statistics
    """
    try:
        if not db_manager:
            raise HTTPException(status_code=503, detail="Database not available")
        
        stats = db_manager.get_statistics()
        return JSONResponse(status_code=200, content=stats)
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'gradcam_initialized': gradcam is not None,
        'database_connected': db_manager is not None
    }


@app.get("/api/status")
async def api_status():
    """Get API status and configuration"""
    return {
        'api_name': 'Document Forgery Detection API',
        'version': '1.0.0',
        'model': 'ResNet50 with Transfer Learning',
        'xai_method': 'Grad-CAM',
        'input_size': INPUT_SIZE,
        'max_batch_size': BATCH_SIZE,
        'supported_formats': ['JPG', 'PNG', 'BMP', 'TIFF', 'PDF'],
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'database': {
            'enabled': db_manager is not None,
            'type': 'MongoDB',
            'url': MONGODB_URL if db_manager else None
        },
        'features': [
            'Binary classification (Authentic vs Tampered)',
            'Grad-CAM heatmap visualization',
            'Error Level Analysis (ELA)',
            'PDF support with page-by-page analysis',
            'Batch processing',
            'MongoDB prediction persistence',
            'Historical prediction queries'
        ],
        'endpoints': {
            'analysis': ['/api/analyze/image', '/api/analyze/pdf', '/api/batch/analyze'],
            'database': ['/api/predictions/recent', '/api/predictions/file/{filename}', '/api/statistics'],
            'system': ['/api/health', '/api/status']
        }
    }


if __name__ == "__main__":
    logger.info("Starting Document Forgery Detection API")
    logger.info(f"Server: http://{API_HOST}:{API_PORT}")
    logger.info("Docs: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD
    )
