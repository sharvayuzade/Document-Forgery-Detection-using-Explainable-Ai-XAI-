"""
Training script for document forgery detection model
Optimized for Intel i5-1220P with limited GPU memory
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import *
from data.data_loader import CASIAv2DataLoader, create_data_loaders
from utils.preprocessing import ImagePreprocessor, create_train_augmentation, create_val_augmentation
from models.cnn_model import DocumentForgeryDetector, CustomCallback
from xai.gradcam import create_gradcam

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocumentForgeryTrainer:
    """Training pipeline for document forgery detection"""
    
    def __init__(self, config):
        """Initialize trainer"""
        self.config = config
        self.preprocessor = ImagePreprocessor(target_size=(config.INPUT_SIZE, config.INPUT_SIZE))
        self.casia_loader, self.pdf_processor = create_data_loaders(config)
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.model = None
        self.history = None
    
    def load_and_split_data(self):
        """Load CASIA v2.0 dataset and split into train/val/test"""
        logger.info("="*80)
        logger.info("STEP 1: Loading CASIA v2.0 Dataset")
        logger.info("="*80)
        
        # Load dataset
        images, labels, paths = self.casia_loader.load_dataset()
        
        if len(images) == 0:
            logger.error("No images loaded. Check dataset paths.")
            return False
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        logger.info(f"Total images: {len(images)}")
        logger.info(f"Authentic: {np.sum(y == 0)}")
        logger.info(f"Tampered: {np.sum(y == 1)}")
        
        # First split: train vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.config.TEST_SPLIT, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = self.config.VAL_SPLIT / (1 - self.config.TEST_SPLIT)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(self.X_train)} ({self.config.TRAIN_SPLIT*100:.1f}%)")
        logger.info(f"  Val:   {len(self.X_val)} ({self.config.VAL_SPLIT*100:.1f}%)")
        logger.info(f"  Test:  {len(self.X_test)} ({self.config.TEST_SPLIT*100:.1f}%)")
        
        return True
    
    def preprocess_data(self, apply_ela: bool = False):
        """Preprocess all data"""
        logger.info("="*80)
        logger.info("STEP 2: Preprocessing Data")
        logger.info("="*80)
        
        logger.info("Preprocessing training data...")
        self.X_train = self.preprocessor.batch_preprocess(
            list(self.X_train), apply_ela=apply_ela, normalize=True
        )
        
        logger.info("Preprocessing validation data...")
        self.X_val = self.preprocessor.batch_preprocess(
            list(self.X_val), apply_ela=False, normalize=True
        )
        
        logger.info("Preprocessing test data...")
        self.X_test = self.preprocessor.batch_preprocess(
            list(self.X_test), apply_ela=False, normalize=True
        )
        
        logger.info(f"\nPreprocessed arrays:")
        logger.info(f"  Train: {self.X_train.shape}")
        logger.info(f"  Val:   {self.X_val.shape}")
        logger.info(f"  Test:  {self.X_test.shape}")
        
        return True
    
    def build_model(self, model_type: str = 'resnet50'):
        """Build and compile model"""
        logger.info("="*80)
        logger.info("STEP 3: Building Model")
        logger.info("="*80)
        
        detector = DocumentForgeryDetector(
            input_shape=(self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3),
            num_classes=self.config.NUM_CLASSES,
            use_mixed_precision=self.config.USE_MIXED_PRECISION
        )
        
        # Build appropriate model
        if model_type.lower() == 'resnet50':
            detector.build_resnet50_model()
        elif model_type.lower() == 'vgg16':
            detector.build_vgg16_model()
        elif model_type.lower() == 'mobilenetv2':
            detector.build_mobilenetv2_model()
        else:
            logger.warning(f"Unknown model {model_type}, using ResNet50")
            detector.build_resnet50_model()
        
        detector.compile_model(learning_rate=self.config.LEARNING_RATE)
        self.model = detector.get_model()
        
        logger.info(f"✓ Model built successfully with {self.model.count_params():,} parameters")
        
        return True
    
    def create_data_generators(self):
        """Create data generators for training"""
        logger.info("Creating data generators...")
        
        train_gen = create_train_augmentation()
        val_gen = create_val_augmentation()
        
        return train_gen, val_gen
    
    def train_model(self, epochs: int = None):
        """Train the model"""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        logger.info("="*80)
        logger.info("STEP 4: Training Model")
        logger.info("="*80)
        
        if self.model is None:
            logger.error("Model must be built first")
            return False
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(self.config.BEST_MODEL_PATH),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(self.config.LOGS_DIR),
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        logger.info(f"\nTraining configuration:")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {self.config.BATCH_SIZE}")
        logger.info(f"  Learning rate: {self.config.LEARNING_RATE}")
        logger.info(f"  Early stopping patience: {self.config.PATIENCE}")
        
        # Train
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=self.config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✓ Training completed")
        return True
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        logger.info("="*80)
        logger.info("STEP 5: Model Evaluation")
        logger.info("="*80)
        
        # Test set evaluation
        test_loss, test_acc, test_prec, test_rec = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        
        logger.info(f"\nTest Set Metrics:")
        logger.info(f"  Loss:      {test_loss:.4f}")
        logger.info(f"  Accuracy:  {test_acc:.4f}")
        logger.info(f"  Precision: {test_prec:.4f}")
        logger.info(f"  Recall:    {test_rec:.4f}")
        
        # Get predictions for detailed metrics
        y_pred = self.model.predict(self.X_test)
        y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(self.y_test, y_pred_binary)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  [[{cm[0,0]:5d} {cm[0,1]:5d}]")
        logger.info(f"   [{cm[1,0]:5d} {cm[1,1]:5d}]]")
        
        logger.info(f"\nDetailed Classification Report:")
        logger.info(classification_report(self.y_test, y_pred_binary, 
                                        target_names=['Authentic', 'Tampered']))
        
        return test_acc, test_loss
    
    def save_model(self):
        """Save trained model"""
        logger.info("="*80)
        logger.info("STEP 6: Saving Model")
        logger.info("="*80)
        
        # Save latest model
        self.model.save(str(self.config.LATEST_MODEL_PATH))
        logger.info(f"✓ Model saved to {self.config.LATEST_MODEL_PATH}")
        
        # Best model is already saved by callback
        if self.config.BEST_MODEL_PATH.exists():
            logger.info(f"✓ Best model saved to {self.config.BEST_MODEL_PATH}")
    
    def run_full_pipeline(self, model_type: str = 'resnet50', apply_ela: bool = False):
        """Run complete training pipeline"""
        try:
            # Step 1: Load data
            if not self.load_and_split_data():
                return False
            
            # Step 2: Preprocess data
            if not self.preprocess_data(apply_ela=apply_ela):
                return False
            
            # Step 3: Build model
            if not self.build_model(model_type=model_type):
                return False
            
            # Step 4: Train model
            if not self.train_model():
                return False
            
            # Step 5: Evaluate model
            self.evaluate_model()
            
            # Step 6: Save model
            self.save_model()
            
            logger.info("\n" + "="*80)
            logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            return True
        
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}", exc_info=True)
            return False


def main():
    """Main entry point"""
    logger.info("\n" + "="*80)
    logger.info("DOCUMENT FORGERY DETECTION - TRAINING PIPELINE")
    logger.info(f"System: Intel i5-1220P with Intel Iris Xe GPU")
    logger.info("="*80 + "\n")
    
    # Create trainer
    trainer = DocumentForgeryTrainer(sys.modules[__name__])
    for attr in dir(sys.modules['src.config']):
        if attr.isupper():
            setattr(trainer.config, attr, getattr(sys.modules['src.config'], attr))
    
    # Use actual config module
    import src.config as config
    trainer.config = config
    
    # Run pipeline
    success = trainer.run_full_pipeline(
        model_type='resnet50',  # Use ResNet50 for balanced performance
        apply_ela=True  # Apply Error Level Analysis
    )
    
    if success:
        logger.info("\n✓ Training successful! Model ready for deployment.")
    else:
        logger.error("\n✗ Training failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
