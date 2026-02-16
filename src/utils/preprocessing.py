"""
Preprocessing module with Error Level Analysis (ELA) for document forgery detection
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ErrorLevelAnalysis:
    """Error Level Analysis for detecting image compression artifacts"""
    
    def __init__(self, quality: int = 90, scale: int = 255):
        """
        Initialize ELA processor
        
        Args:
            quality: JPEG compression quality (1-100)
            scale: Scale factor for visualization
        """
        self.quality = quality
        self.scale = scale
    
    def compute_ela(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Error Level Analysis
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            ELA heatmap as grayscale image
        """
        try:
            # Convert BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Save at specified quality and reload
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            compressed_image = Image.open(buffer)
            compressed_array = np.array(compressed_image)
            
            # Convert back to BGR
            compressed_image_bgr = cv2.cvtColor(compressed_array, cv2.COLOR_RGB2BGR)
            
            # Calculate absolute difference
            diff = cv2.absdiff(image.astype(np.float32), compressed_image_bgr.astype(np.float32))
            
            # Scale the difference
            ela_image = (diff * self.scale / 255.0).astype(np.uint8)
            
            # Convert to grayscale for better visualization
            ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
            
            return ela_gray
        
        except Exception as e:
            logger.error(f"Error in ELA computation: {e}")
            return np.zeros_like(image[:, :, 0])
    
    def apply_adaptive_histogram_equalization(self, image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image
            clip_limit: Contrast limit
            
        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Convert BGR to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            
            # Merge back
            lab_clahe = cv2.merge([l_clahe, a, b])
            enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced


class ImagePreprocessor:
    """Handle image preprocessing and augmentation"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image preprocessor
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        self.ela = ErrorLevelAnalysis()
    
    def resize_image(self, image: np.ndarray, maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        if maintain_aspect:
            # Calculate aspect ratio
            aspect = w / h
            target_aspect = target_w / target_h
            
            if aspect > target_aspect:
                # Image is wider
                new_w = target_w
                new_h = int(target_w / aspect)
            else:
                # Image is taller
                new_h = target_h
                new_w = int(target_h * aspect)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad to target size
            padded = np.zeros((target_h, target_w, 3 if len(image.shape) == 3 else 1), dtype=image.dtype)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        else:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    def normalize_image(self, image: np.ndarray, mean: List[float] = None, 
                       std: List[float] = None) -> np.ndarray:
        """
        Normalize image using ImageNet parameters
        
        Args:
            image: Input image in BGR format
            mean: Mean values for each channel
            std: Standard deviation values for each channel
            
        Returns:
            Normalized image
        """
        if mean is None:
            # ImageNet means in BGR
            mean = [0.406, 0.456, 0.485]
        if std is None:
            # ImageNet standard deviations in BGR
            std = [0.225, 0.224, 0.229]
        
        # Normalize
        normalized = image.astype(np.float32) / 255.0
        normalized[:, :, 0] = (normalized[:, :, 0] - mean[0]) / std[0]
        normalized[:, :, 1] = (normalized[:, :, 1] - mean[1]) / std[1]
        normalized[:, :, 2] = (normalized[:, :, 2] - mean[2]) / std[2]
        
        return normalized
    
    def preprocess_image(self, image: np.ndarray, apply_ela: bool = False, 
                        normalize: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image
            apply_ela: Whether to apply Error Level Analysis
            normalize: Whether to normalize image
            
        Returns:
            Preprocessed image
        """
        # Convert to BGR if necessary
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Resize
        resized = self.resize_image(image)
        
        # Apply ELA if requested
        if apply_ela:
            ela = self.ela.compute_ela(resized)
            # Stack ELA with original image
            resized = np.dstack([resized, np.stack([ela, ela, ela], axis=2)])
        
        # Enhance contrast
        enhanced = self.ela.apply_adaptive_histogram_equalization(resized)
        
        # Normalize
        if normalize:
            normalized = self.normalize_image(enhanced)
        else:
            normalized = enhanced.astype(np.float32) / 255.0
        
        return normalized
    
    def batch_preprocess(self, images: List[np.ndarray], apply_ela: bool = False,
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess batch of images
        
        Args:
            images: List of images
            apply_ela: Whether to apply ELA
            normalize: Whether to normalize
            
        Returns:
            Batch of preprocessed images (numpy array)
        """
        processed = []
        
        for i, image in enumerate(images):
            try:
                preprocessed = self.preprocess_image(image, apply_ela, normalize)
                processed.append(preprocessed)
            except Exception as e:
                logger.warning(f"Error preprocessing image {i}: {e}")
                continue
        
        return np.array(processed)


def create_train_augmentation():
    """Create data augmentation for training"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect'
    )


def create_val_augmentation():
    """Create minimal augmentation for validation/testing"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    return ImageDataGenerator(
        fill_mode='reflect'
    )


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ImagePreprocessor()
    ela = ErrorLevelAnalysis()
    
    # Create test image
    test_img = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
    
    # Test ELA
    ela_result = ela.compute_ela(test_img)
    print(f"ELA result shape: {ela_result.shape}")
    
    # Test preprocessing
    processed = preprocessor.preprocess_image(test_img, apply_ela=True, normalize=True)
    print(f"Processed image shape: {processed.shape}")
    print(f"Processed image dtype: {processed.dtype}")
    print(f"Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
