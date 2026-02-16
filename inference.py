"""
Standalone inference script for batch processing documents
Generates predictions, heatmaps, and descriptions for each input
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
import json
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import *
from data.data_loader import PDFProcessor
from utils.preprocessing import ImagePreprocessor
from xai.gradcam import GradCAM
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentForgertyInference:
    """Inference pipeline for document forgery detection"""
    
    def __init__(self, model_path: str = None, device: str = 'GPU'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model
            device: Device to use ('GPU' or 'CPU')
        """
        self.device = device
        self.model = None
        self.gradcam = None
        self.preprocessor = None
        self.pdf_processor = None
        
        self._setup_gpu()
        self._load_model(model_path)
    
    def _setup_gpu(self):
        """Setup GPU if available"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and self.device == 'GPU':
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"✓ GPU available: {gpus}")
            else:
                logger.info("⚠ GPU not available, using CPU")
                self.device = 'CPU'
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
            self.device = 'CPU'
    
    def _load_model(self, model_path: str = None):
        """Load trained model"""
        if model_path is None:
            model_path = str(BEST_MODEL_PATH)
        
        logger.info(f"Loading model from {model_path}...")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"✓ Model loaded: {self.model.name}")
        
        # Find layer for Grad-CAM
        layer_name = None
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
        
        if layer_name:
            self.gradcam = GradCAM(self.model, layer_name)
            logger.info(f"✓ Grad-CAM initialized with layer: {layer_name}")
        
        # Initialize other components
        self.preprocessor = ImagePreprocessor(target_size=(INPUT_SIZE, INPUT_SIZE))
        self.pdf_processor = PDFProcessor(dpi=PDF_DPI)
    
    def predict_image(self, image_path: str) -> dict:
        """
        Predict on single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with predictions and analysis
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            logger.info(f"Processing: {Path(image_path).name}")
            
            # Preprocess
            preprocessed = self.preprocessor.preprocess_image(image, apply_ela=False, normalize=True)
            preprocessed = np.expand_dims(preprocessed, axis=0)
            
            # Predict
            prediction = self.model.predict(preprocessed, verbose=0)[0][0]
            
            # Generate Grad-CAM
            heatmap = None
            overlay = None
            if self.gradcam:
                heatmap = self.gradcam.compute_gradcam(preprocessed)[0]
                overlay = self.gradcam.overlay_heatmap(image, heatmap, alpha=HEATMAP_ALPHA)
            
            # Generate report
            is_tampered = prediction > CONFIDENCE_THRESHOLD
            confidence = prediction if is_tampered else 1.0 - prediction
            
            result = {
                'file': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'prediction_score': float(prediction),
                'is_tampered': bool(is_tampered),
                'status': 'TAMPERED' if is_tampered else 'AUTHENTIC',
                'confidence': float(confidence),
                'confidence_percentage': float(confidence * 100),
                'forgery_score': float(np.mean(heatmap)) if heatmap is not None else None,
                'image': image,
                'heatmap': heatmap,
                'overlay': overlay
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'file': str(image_path),
                'error': str(e)
            }
    
    def predict_pdf(self, pdf_path: str) -> List[dict]:
        """
        Predict on PDF document (all pages)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of predictions for each page
        """
        try:
            logger.info(f"Processing PDF: {Path(pdf_path).name}")
            
            # Extract images
            images = self.pdf_processor.pdf_to_images(pdf_path)
            
            if not images:
                raise ValueError("Could not extract images from PDF")
            
            logger.info(f"Extracted {len(images)} pages from PDF")
            
            results = []
            for page_num, image in enumerate(images):
                try:
                    # Preprocess
                    preprocessed = self.preprocessor.preprocess_image(image, apply_ela=False, normalize=True)
                    preprocessed = np.expand_dims(preprocessed, axis=0)
                    
                    # Predict
                    prediction = self.model.predict(preprocessed, verbose=0)[0][0]
                    
                    # Grad-CAM
                    heatmap = None
                    overlay = None
                    if self.gradcam:
                        heatmap = self.gradcam.compute_gradcam(preprocessed)[0]
                        overlay = self.gradcam.overlay_heatmap(image, heatmap, alpha=HEATMAP_ALPHA)
                    
                    is_tampered = prediction > CONFIDENCE_THRESHOLD
                    confidence = prediction if is_tampered else 1.0 - prediction
                    
                    result = {
                        'file': str(pdf_path),
                        'page': page_num + 1,
                        'timestamp': datetime.now().isoformat(),
                        'prediction_score': float(prediction),
                        'is_tampered': bool(is_tampered),
                        'status': 'TAMPERED' if is_tampered else 'AUTHENTIC',
                        'confidence': float(confidence),
                        'confidence_percentage': float(confidence * 100),
                        'forgery_score': float(np.mean(heatmap)) if heatmap is not None else None,
                        'image': image,
                        'heatmap': heatmap,
                        'overlay': overlay
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return [{'file': str(pdf_path), 'error': str(e)}]
    
    def save_results(self, results: List[dict], output_dir: str = None):
        """
        Save analysis results and visualizations
        
        Args:
            results: List of analysis results
            output_dir: Output directory for results
        """
        if output_dir is None:
            output_dir = OUTPUTS_DIR
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / 'heatmaps').mkdir(exist_ok=True)
        (output_dir / 'overlays').mkdir(exist_ok=True)
        (output_dir / 'masks').mkdir(exist_ok=True)
        (output_dir / 'reports').mkdir(exist_ok=True)
        
        # Save visualizations and generate JSON report
        json_results = []
        
        for i, result in enumerate(results):
            if 'error' in result:
                json_results.append(result)
                continue
            
            # Extract file info
            file_name = Path(result['file']).stem
            page = result.get('page', 1)
            
            # Generate output filename
            if 'page' in result:
                prefix = f"{file_name}_page{page}"
            else:
                prefix = file_name
            
            # Save heatmap
            if result.get('heatmap') is not None:
                heatmap_path = output_dir / 'heatmaps' / f"{prefix}_heatmap.png"
                heatmap_uint8 = (result['heatmap'] * 255).astype(np.uint8)
                cv2.imwrite(str(heatmap_path), heatmap_uint8)
            
            # Save overlay
            if result.get('overlay') is not None:
                overlay_path = output_dir / 'overlays' / f"{prefix}_overlay.png"
                cv2.imwrite(str(overlay_path), result['overlay'])
            
            # Prepare JSON entry
            json_entry = {k: v for k, v in result.items() 
                         if k not in ['image', 'heatmap', 'overlay']}
            json_entry['heatmap_file'] = f"heatmaps/{prefix}_heatmap.png"
            json_entry['overlay_file'] = f"overlays/{prefix}_overlay.png"
            json_results.append(json_entry)
        
        # Save JSON report
        report_path = output_dir / 'reports' / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"✓ Results saved to {output_dir}")
        logger.info(f"✓ Report saved to {report_path}")
        
        return str(report_path)
    
    def batch_process(self, input_dir: str, output_dir: str = None, 
                     file_pattern: str = '*.*') -> str:
        """
        Batch process all files in directory
        
        Args:
            input_dir: Directory containing images/PDFs
            output_dir: Output directory for results
            file_pattern: File pattern to match
            
        Returns:
            Path to report
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        logger.info(f"Processing directory: {input_dir}")
        
        all_results = []
        
        # Process files
        files = sorted(input_path.glob(file_pattern))
        logger.info(f"Found {len(files)} files")
        
        for i, file_path in enumerate(files, 1):
            logger.info(f"[{i}/{len(files)}] {file_path.name}")
            
            if file_path.suffix.lower() == '.pdf':
                results = self.predict_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                results = [self.predict_image(str(file_path))]
            else:
                continue
            
            all_results.extend(results)
        
        # Save results
        report_path = self.save_results(all_results, output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Total items processed: {len(all_results)}")
        print(f"Authentic: {sum(1 for r in all_results if r.get('status') == 'AUTHENTIC')}")
        print(f"Tampered: {sum(1 for r in all_results if r.get('status') == 'TAMPERED')}")
        print(f"Report saved: {report_path}")
        print("="*80)
        
        return report_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Document Forgery Detection Inference"
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input file or directory')
    parser.add_argument('--output', '-o', default=None,
                       help='Output directory (default: outputs/)')
    parser.add_argument('--model', '-m', default=None,
                       help='Model path (default: models/best_model.h5)')
    parser.add_argument('--device', '-d', default='GPU', choices=['GPU', 'CPU'],
                       help='Device to use')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process directory in batch mode')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("DOCUMENT FORGERY DETECTION - INFERENCE")
    logger.info("="*80)
    
    try:
        # Initialize inference engine
        inference = DocumentForgertyInference(
            model_path=args.model,
            device=args.device
        )
        
        input_path = Path(args.input)
        
        if args.batch or input_path.is_dir():
            # Batch processing
            inference.batch_process(args.input, args.output)
        else:
            # Single file processing
            if input_path.suffix.lower() == '.pdf':
                results = inference.predict_pdf(args.input)
            else:
                results = [inference.predict_image(args.input)]
            
            # Save results
            inference.save_results(results, args.output)
            
            # Print summary
            if results and 'error' not in results[0]:
                result = results[0]
                print("\n" + "="*80)
                print("ANALYSIS RESULT")
                print("="*80)
                print(f"File: {result['file']}")
                print(f"Status: {result['status']}")
                print(f"Confidence: {result['confidence_percentage']:.2f}%")
                print(f"Forgery Score: {result['forgery_score']:.4f}" if result['forgery_score'] else "")
                print("="*80)
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
