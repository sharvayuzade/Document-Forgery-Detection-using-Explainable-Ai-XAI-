"""
CNN Model architecture with Transfer Learning for document forgery detection
Optimized for Intel i5-1220P with limited GPU memory
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class DocumentForgeryDetector:
    """Transfer learning model for document forgery detection"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3), 
                 num_classes: int = 2, use_mixed_precision: bool = True):
        """
        Initialize document forgery detector model
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes (2 for binary classification)
            use_mixed_precision: Whether to use mixed precision training
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_mixed_precision = use_mixed_precision
        
        # Set up mixed precision if requested
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("✓ Mixed precision training enabled")
        
        self.model = None
        self.base_model = None
    
    def build_resnet50_model(self) -> keras.Model:
        """
        Build ResNet50 transfer learning model
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building ResNet50 model...")
        
        # Load pretrained ResNet50
        self.base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        for layer in self.base_model.layers:
            layer.trainable = False
        
        logger.info(f"✓ Base model layers: {len(self.base_model.layers)}")
        
        # Build custom top layers
        x = self.base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        if self.num_classes == 2:
            output = layers.Dense(1, activation='sigmoid')(x)
        else:
            output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=self.base_model.input, outputs=output)
        
        return self.model
    
    def build_vgg16_model(self) -> keras.Model:
        """
        Build VGG16 transfer learning model (lighter alternative)
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building VGG16 model...")
        
        # Load pretrained VGG16
        self.base_model = keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Build custom top layers
        x = self.base_model.output
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        if self.num_classes == 2:
            output = layers.Dense(1, activation='sigmoid')(x)
        else:
            output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=self.base_model.input, outputs=output)
        
        return self.model
    
    def build_mobilenetv2_model(self) -> keras.Model:
        """
        Build MobileNetV2 transfer learning model (most lightweight)
        Best for Intel i5-1220P with limited resources
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building MobileNetV2 model (optimized for limited resources)...")
        
        # Load pretrained MobileNetV2
        self.base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Build custom top layers (minimal)
        x = self.base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        if self.num_classes == 2:
            output = layers.Dense(1, activation='sigmoid')(x)
        else:
            output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=self.base_model.input, outputs=output)
        
        return self.model
    
    def compile_model(self, learning_rate: float = 0.001, optimizer: str = 'adam'):
        """
        Compile the model with loss and optimizer
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        # Choose optimizer
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        elif optimizer.lower() == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile with appropriate loss
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        else:
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"✓ Model compiled with {optimizer} optimizer (lr={learning_rate})")
        logger.info(f"✓ Model summary:")
        self.model.summary()
    
    def unfreeze_layers(self, num_layers: int):
        """
        Unfreeze last N layers for fine-tuning
        
        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        if self.base_model is None:
            raise ValueError("Base model must be built first")
        
        for layer in self.base_model.layers[-num_layers:]:
            layer.trainable = True
        
        logger.info(f"✓ Unfroze last {num_layers} layers for fine-tuning")
    
    def get_model(self) -> keras.Model:
        """Get the compiled model"""
        if self.model is None:
            raise ValueError("Model must be built first")
        return self.model
    
    def get_base_model(self) -> keras.Model:
        """Get the base model (for Grad-CAM and other XAI techniques)"""
        if self.base_model is None:
            raise ValueError("Base model must be built first")
        return self.base_model


class CustomCallback(keras.callbacks.Callback):
    """Custom callback for training monitoring optimized for limited resources"""
    
    def __init__(self, log_dir: str, patience: int = 5, min_delta: float = 0.0001):
        """
        Initialize custom callback
        
        Args:
            log_dir: Directory for saving logs
            patience: Patience for early stopping
            min_delta: Minimum change to qualify as improvement
        """
        super().__init__()
        self.log_dir = log_dir
        self.patience = patience
        self.min_delta = min_delta
        self.wait_count = 0
        self.best_val_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        logs = logs or {}
        
        # Check if validation loss improved
        val_loss = logs.get('val_loss')
        if val_loss is not None:
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.wait_count = 0
                logger.info(f"✓ Epoch {epoch}: Validation loss improved to {val_loss:.4f}")
            else:
                self.wait_count += 1
                if self.wait_count >= self.patience:
                    logger.info(f"✗ Early stopping triggered after {self.patience} epochs without improvement")
                    self.model.stop_training = True


def create_model(model_type: str = 'resnet50', input_shape: Tuple[int, int, int] = (224, 224, 3),
                num_classes: int = 2, learning_rate: float = 0.001, 
                use_mixed_precision: bool = True) -> keras.Model:
    """
    Factory function to create and compile model
    
    Args:
        model_type: Type of model ('resnet50', 'vgg16', 'mobilenetv2')
        input_shape: Input image shape
        num_classes: Number of classes
        learning_rate: Learning rate
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Compiled Keras model
    """
    detector = DocumentForgeryDetector(input_shape, num_classes, use_mixed_precision)
    
    if model_type.lower() == 'resnet50':
        detector.build_resnet50_model()
    elif model_type.lower() == 'vgg16':
        detector.build_vgg16_model()
    elif model_type.lower() == 'mobilenetv2':
        detector.build_mobilenetv2_model()
    else:
        logger.warning(f"Unknown model type {model_type}, using ResNet50")
        detector.build_resnet50_model()
    
    detector.compile_model(learning_rate=learning_rate)
    
    return detector.get_model()


if __name__ == "__main__":
    # Test model creation
    model = create_model(model_type='mobilenetv2', learning_rate=0.001)
    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {model.count_params():,}")
