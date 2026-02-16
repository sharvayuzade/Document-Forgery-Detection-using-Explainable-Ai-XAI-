"""
Data loading utilities for CASIA v2.0 dataset and PDF handling
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import List, Tuple, Optional
import io

logger = logging.getLogger(__name__)

try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("pdf2image not installed. PDF support disabled.")

try:
    import fitz  # PyMuPDF
    FITZ_SUPPORT = True
except ImportError:
    FITZ_SUPPORT = False
    logger.warning("PyMuPDF not installed. Using alternative PDF support.")


class CASIAv2DataLoader:
    """Load CASIA v2.0 dataset with Au (Authentic) and Tp (Tampered) folders"""
    
    def __init__(self, authentic_dir: str, tampered_dir: str, groundtruth_dir: str = None):
        """
        Initialize CASIA v2.0 data loader
        
        Args:
            authentic_dir: Path to authentic images folder
            tampered_dir: Path to tampered images folder
            groundtruth_dir: Path to ground truth masks (optional)
        """
        self.authentic_dir = Path(authentic_dir)
        self.tampered_dir = Path(tampered_dir)
        self.groundtruth_dir = Path(groundtruth_dir) if groundtruth_dir else None
        
        if not self.authentic_dir.exists():
            raise FileNotFoundError(f"Authentic directory not found: {authentic_dir}")
        if not self.tampered_dir.exists():
            raise FileNotFoundError(f"Tampered directory not found: {tampered_dir}")
        
        logger.info(f"✓ CASIA v2.0 loader initialized")
        logger.info(f"  Authentic dir: {self.authentic_dir}")
        logger.info(f"  Tampered dir: {self.tampered_dir}")
    
    def get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files from directory"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        files = []
        
        for ext in valid_extensions:
            files.extend(directory.glob(f'**/*{ext}'))
            files.extend(directory.glob(f'**/*{ext.upper()}'))
        
        return sorted(list(set(files)))  # Remove duplicates and sort
    
    def load_dataset(self) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Load complete CASIA v2.0 dataset
        
        Returns:
            images: List of image arrays
            labels: List of labels (0=Authentic, 1=Tampered)
            image_paths: List of image file paths
        """
        images = []
        labels = []
        image_paths = []
        
        # Load Authentic images
        logger.info("Loading Authentic images...")
        au_files = self.get_image_files(self.authentic_dir)
        for file_path in au_files:
            try:
                img = cv2.imread(str(file_path))
                if img is not None:
                    images.append(img)
                    labels.append(0)  # Authentic
                    image_paths.append(str(file_path))
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        logger.info(f"✓ Loaded {len([l for l in labels if l == 0])} authentic images")
        
        # Load Tampered images
        logger.info("Loading Tampered images...")
        tp_files = self.get_image_files(self.tampered_dir)
        for file_path in tp_files:
            try:
                img = cv2.imread(str(file_path))
                if img is not None:
                    images.append(img)
                    labels.append(1)  # Tampered
                    image_paths.append(str(file_path))
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
        
        logger.info(f"✓ Loaded {len([l for l in labels if l == 1])} tampered images")
        logger.info(f"✓ Total images loaded: {len(images)}")
        
        return images, labels, image_paths
    
    def load_groundtruth(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load ground truth mask for tampered image
        
        Args:
            image_path: Path to tampered image
            
        Returns:
            Ground truth mask as numpy array or None
        """
        if self.groundtruth_dir is None:
            return None
        
        try:
            filename = Path(image_path).stem
            mask_path = self.groundtruth_dir / f"{filename}_gt.png"
            
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                return mask
        except Exception as e:
            logger.warning(f"Error loading ground truth for {image_path}: {e}")
        
        return None


class PDFProcessor:
    """Handle PDF to image conversion"""
    
    def __init__(self, dpi: int = 150, max_pages: int = 50):
        """
        Initialize PDF processor
        
        Args:
            dpi: Resolution for PDF to image conversion
            max_pages: Maximum number of pages to process
        """
        self.dpi = dpi
        self.max_pages = max_pages
    
    def is_pdf(self, file_path: str) -> bool:
        """Check if file is a PDF"""
        return str(file_path).lower().endswith('.pdf')
    
    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF to list of images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of image arrays (BGR format from OpenCV)
        """
        images = []
        
        try:
            # Try PyMuPDF first (faster)
            if FITZ_SUPPORT:
                images = self._pdf_to_images_fitz(pdf_path)
            elif PDF_SUPPORT:
                images = self._pdf_to_images_pdf2image(pdf_path)
            else:
                logger.error("No PDF library available. Install pdf2image or PyMuPDF.")
                return []
        
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {e}")
            return []
        
        logger.info(f"✓ Converted PDF to {len(images)} images")
        return images
    
    def _pdf_to_images_fitz(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF using PyMuPDF (faster)"""
        images = []
        
        try:
            pdf_document = fitz.open(pdf_path)
            num_pages = min(len(pdf_document), self.max_pages)
            
            for page_num in range(num_pages):
                page = pdf_document[page_num]
                # Higher zoom for better quality
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to OpenCV format (BGR)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                # Convert RGB to BGR for OpenCV
                if pix.n == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif pix.n == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                
                images.append(img_array)
        
        except Exception as e:
            raise e
        
        return images
    
    def _pdf_to_images_pdf2image(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF using pdf2image"""
        images = []
        
        try:
            pil_images = pdf2image.convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=1,
                last_page=self.max_pages
            )
            
            for pil_img in pil_images:
                # Convert PIL to OpenCV (BGR)
                img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                images.append(img_array)
        
        except Exception as e:
            raise e
        
        return images
    
    def process_input_file(self, file_path: str) -> List[np.ndarray]:
        """
        Process input file (PDF or Image)
        
        Args:
            file_path: Path to input file
            
        Returns:
            List of image arrays
        """
        if self.is_pdf(file_path):
            return self.pdf_to_images(file_path)
        else:
            # Load as single image
            img = cv2.imread(str(file_path))
            if img is not None:
                return [img]
            else:
                logger.error(f"Could not load image: {file_path}")
                return []


def create_data_loaders(config):
    """Factory function to create data loaders with config"""
    casia_loader = CASIAv2DataLoader(
        str(config.AUTHENTIC_DIR),
        str(config.TAMPERED_DIR),
        str(config.GROUNDTRUTH_DIR) if config.GROUNDTRUTH_DIR else None
    )
    
    pdf_processor = PDFProcessor(
        dpi=config.PDF_DPI,
        max_pages=config.PDF_MAX_PAGES
    )
    
    return casia_loader, pdf_processor


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import AUTHENTIC_DIR, TAMPERED_DIR, GROUNDTRUTH_DIR
    
    # Test CASIA loader
    loader = CASIAv2DataLoader(str(AUTHENTIC_DIR), str(TAMPERED_DIR), str(GROUNDTRUTH_DIR))
    images, labels, paths = loader.load_dataset()
    print(f"Loaded {len(images)} images")
    print(f"Authentic: {sum(1 for l in labels if l == 0)}")
    print(f"Tampered: {sum(1 for l in labels if l == 1)}")
