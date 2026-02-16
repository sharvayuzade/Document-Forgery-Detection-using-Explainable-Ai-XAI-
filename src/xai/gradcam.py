"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
For explaining CNN predictions in document forgery detection
"""

import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretation"""
    
    def __init__(self, model: tf.keras.Model, layer_name: str):
        """
        Initialize Grad-CAM
        
        Args:
            model: Keras model
            layer_name: Name of the layer to compute gradients for
        """
        self.model = model
        self.layer_name = layer_name
        
        # Find the layer
        self.layer = None
        for layer in model.layers:
            if layer.name == layer_name:
                self.layer = layer
                break
        
        if self.layer is None:
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Create gradient model
        self.grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[self.layer.output, model.output]
        )
        
        logger.info(f"✓ Grad-CAM initialized with layer: {layer_name}")
    
    def compute_gradcam(self, images: np.ndarray, class_idx: int = None) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for input images
        
        Args:
            images: Input images array (batch or single image)
            class_idx: Class index to compute gradients for (None for predicted class)
            
        Returns:
            Grad-CAM heatmaps (same spatial size as original images)
        """
        # Handle single image
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        batch_size = images.shape[0]
        original_height, original_width = images.shape[1:3]
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(images)
            
            # Determine class index
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            class_channel = predictions[:, class_idx]
        
        # Get gradients of the class with respect to the conv layer output
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Average over spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        
        # Compute weighted feature maps (Grad-CAM)
        conv_outputs = conv_outputs.numpy()
        pooled_grads = pooled_grads.numpy()
        
        heatmaps = []
        for i in range(batch_size):
            # Weight the conv layer output channels
            weighted_conv = conv_outputs[i]
            for j in range(weighted_conv.shape[-1]):
                weighted_conv[:, :, j] *= pooled_grads[i, j]
            
            # Average over channels
            heatmap = np.mean(weighted_conv, axis=2)
            
            # Apply ReLU to keep only features that increase class probability
            heatmap = np.maximum(heatmap, 0)
            
            # Normalize to 0-1
            heatmap_max = np.max(heatmap)
            if heatmap_max > 0:
                heatmap = heatmap / heatmap_max
            
            # Resize to original image size
            heatmap_resized = cv2.resize(heatmap, (original_width, original_height))
            heatmaps.append(heatmap_resized)
        
        return np.array(heatmaps)
    
    def overlay_heatmap(self, image: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image
        
        Args:
            image: Original image (BGR format from OpenCV)
            heatmap: Grad-CAM heatmap (0-1 normalized)
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap to use
            
        Returns:
            Image with heatmap overlay
        """
        # Convert heatmap to 0-255 range
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
        
        # Resize heatmap to match image size if necessary
        if heatmap_color.shape[:2] != image.shape[:2]:
            heatmap_color = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
        
        # Blend image and heatmap
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
        
        return overlay
    
    def generate_report(self, image: np.ndarray, prediction: float, 
                       heatmap: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Generate detailed analysis report
        
        Args:
            image: Original image
            prediction: Model prediction (0-1 confidence)
            heatmap: Grad-CAM heatmap
            threshold: Decision threshold
            
        Returns:
            Dictionary with analysis results
        """
        # Get overlay
        overlay = self.overlay_heatmap(image, heatmap)
        
        # Analyze heatmap statistics
        high_attention_pixels = np.sum(heatmap > 0.5)
        total_pixels = heatmap.shape[0] * heatmap.shape[1]
        forgery_score = np.mean(heatmap)
        
        # Determine prediction
        is_tampered = prediction > threshold
        confidence = prediction if is_tampered else 1.0 - prediction
        
        # Create suspicious regions mask
        suspicious_mask = (heatmap > 0.5).astype(np.uint8) * 255
        
        report = {
            'prediction': float(prediction),
            'is_tampered': bool(is_tampered),
            'confidence': float(confidence),
            'forgery_score': float(forgery_score),
            'suspicious_pixels': int(high_attention_pixels),
            'total_pixels': int(total_pixels),
            'suspicious_percentage': float(100 * high_attention_pixels / total_pixels),
            'overlay_image': overlay,
            'heatmap': heatmap,
            'suspicious_mask': suspicious_mask
        }
        
        return report


class MultiLayerGradCAM:
    """Compute Grad-CAM from multiple layers for comprehensive analysis"""
    
    def __init__(self, model: tf.keras.Model, layer_names: List[str]):
        """
        Initialize multi-layer Grad-CAM
        
        Args:
            model: Keras model
            layer_names: List of layer names to compute Grad-CAM for
        """
        self.model = model
        self.gradcams = {}
        
        for layer_name in layer_names:
            try:
                self.gradcams[layer_name] = GradCAM(model, layer_name)
            except ValueError as e:
                logger.warning(f"Could not initialize Grad-CAM for layer {layer_name}: {e}")
    
    def compute_all(self, images: np.ndarray) -> dict:
        """
        Compute Grad-CAM for all layers
        
        Args:
            images: Input images
            
        Returns:
            Dictionary with heatmaps for each layer
        """
        results = {}
        for layer_name, gradcam in self.gradcams.items():
            results[layer_name] = gradcam.compute_gradcam(images)
        return results


def find_layer_by_pattern(model: tf.keras.Model, pattern: str) -> str:
    """
    Find layer name matching pattern
    
    Args:
        model: Keras model
        pattern: String pattern to match
        
    Returns:
        First matching layer name
    """
    for layer in model.layers:
        if pattern in layer.name:
            return layer.name
    
    raise ValueError(f"No layer found matching pattern: {pattern}")


def get_model_architecture_info(model: tf.keras.Model):
    """
    Get information about model architecture for Grad-CAM selection
    
    Args:
        model: Keras model
    """
    logger.info("Model layer information:")
    logger.info("=" * 80)
    
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if hasattr(layer, 'output_shape'):
            logger.info(f"{i:3d}. {layer.name:40s} {layer_type:20s} {str(layer.output_shape):30s}")
        else:
            logger.info(f"{i:3d}. {layer.name:40s} {layer_type:20s}")
    
    logger.info("=" * 80)


# Convenience function to create Grad-CAM from model
def create_gradcam(model: tf.keras.Model, layer_name: str = None) -> GradCAM:
    """
    Create Grad-CAM from model (auto-detect best layer if not specified)
    
    Args:
        model: Keras model
        layer_name: Layer name (optional, auto-detected if None)
        
    Returns:
        Initialized GradCAM object
    """
    if layer_name is None:
        # Auto-detect best layer (usually last convolutional layer before pooling)
        layer_name = find_layer_by_pattern(model, 'conv')
    
    return GradCAM(model, layer_name)


if __name__ == "__main__":
    # Test Grad-CAM with a simple model
    from tensorflow.keras.applications import ResNet50
    
    logger.basicConfig(level=logging.INFO)
    
    # Load model
    model = ResNet50(weights='imagenet', include_top=True)
    
    # Find appropriate layer
    conv_layer = find_layer_by_pattern(model, 'conv5_block3_3')
    logger.info(f"Using layer: {conv_layer}")
    
    # Create Grad-CAM
    gradcam = GradCAM(model, conv_layer)
    
    # Create dummy image
    dummy_image = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # Compute heatmap
    heatmap = gradcam.compute_gradcam(dummy_image)
    logger.info(f"Heatmap shape: {heatmap.shape}")
    logger.info(f"Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
