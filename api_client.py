"""
API Client for Document Forgery Detection System
For testing and interacting with the FastAPI backend
"""

import requests
import json
import argparse
from pathlib import Path
from typing import List, Dict
import base64
import io
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForgertyDetectionAPIClient:
    """Client for Document Forgery Detection API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of API server
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ API Status: {data['status']}")
                print(f"  Model loaded: {data['model_loaded']}")
                print(f"  Grad-CAM initialized: {data['gradcam_initialized']}")
                return True
            else:
                print(f"✗ API error: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print(f"✗ Could not connect to API at {self.base_url}")
            return False
    
    def get_api_status(self) -> Dict:
        """Get API status and configuration"""
        try:
            response = self.session.get(f"{self.base_url}/api/status")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"✗ Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Analysis result
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"✗ File not found: {image_path}")
            return None
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (image_path.name, f)}
                response = self.session.post(
                    f"{self.base_url}/api/analyze/image",
                    files=files
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"✗ Error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def analyze_pdf(self, pdf_path: str) -> Dict:
        """
        Analyze PDF document
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Analysis result with all pages
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            print(f"✗ File not found: {pdf_path}")
            return None
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'file': (pdf_path.name, f)}
                response = self.session.post(
                    f"{self.base_url}/api/analyze/pdf",
                    files=files,
                    timeout=300  # 5 minutes for large PDFs
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"✗ Error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            print(f"✗ Error: {e}")
            return None
    
    def batch_analyze(self, file_paths: List[str]) -> Dict:
        """
        Analyze multiple files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Analysis results
        """
        files = []
        
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"⚠ File not found: {path}")
                continue
            
            files.append(('files', open(file_path, 'rb')))
        
        if not files:
            print("✗ No valid files provided")
            return None
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/batch/analyze",
                files=files,
                timeout=600  # 10 minutes for batch
            )
            
            for _, f in files:
                f.close()
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"✗ Error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            print(f"✗ Error: {e}")
            for _, f in files:
                f.close()
            return None
    
    def print_results(self, results: Dict, save_images: bool = False, output_dir: str = None):
        """
        Pretty print results
        
        Args:
            results: Analysis results
            save_images: Whether to save base64 images to files
            output_dir: Directory to save images
        """
        if results is None:
            return
        
        # Single image result
        if 'prediction_score' in results:
            print("\n" + "="*80)
            print("ANALYSIS RESULT")
            print("="*80)
            print(f"Status: {results.get('analysis', {}).get('status', 'N/A')}")
            print(f"Prediction Score: {results.get('prediction_score', 0):.4f}")
            print(f"Confidence: {results.get('analysis', {}).get('confidence_percentage', 0):.2f}%")
            print(f"Forgery Score: {results.get('forgery_score', 0):.4f}")
            print(f"Suspicious Percentage: {results.get('suspicious_percentage', 0):.2f}%")
            print("="*80)
            
            if save_images and output_dir:
                self._save_base64_images(results, output_dir)
        
        # PDF results
        elif 'pages' in results:
            print("\n" + "="*80)
            print("PDF ANALYSIS RESULT")
            print("="*80)
            print(f"Total Pages: {results.get('total_pages', 0)}")
            print(f"Authentic Pages: {results.get('summary', {}).get('authentic_pages', 0)}")
            print(f"Tampered Pages: {results.get('summary', {}).get('tampered_pages', 0)}")
            print(f"Integrity: {results.get('integrity_status', 'N/A')}")
            print("="*80)
            
            pages = results.get('pages', [])
            for page in pages:
                if 'error' in page:
                    print(f"Page {page.get('page_number', '?')}: ERROR - {page['error']}")
                else:
                    print(f"Page {page.get('page_number', '?')}: {page.get('status', 'N/A')} "
                          f"(Score: {page.get('prediction_score', 0):.4f})")
            
            if save_images and output_dir:
                self._save_base64_images(results, output_dir,is_pdf=True)
        
        # Batch results
        elif 'results' in results:
            print("\n" + "="*80)
            print("BATCH ANALYSIS RESULTS")
            print("="*80)
            
            batch_results = results.get('results', [])
            print(f"Total items: {len(batch_results)}")
            
            for item in batch_results:
                if 'error' in item:
                    print(f"  ✗ {item.get('file')}: {item['error']}")
                else:
                    status = "✓" if item.get('status') == 'AUTHENTIC' else "⚠"
                    print(f"  {status} {item.get('file')}: {item.get('status')} "
                          f"(Score: {item.get('prediction_score', 0):.4f})")
            print("="*80)
    
    def _save_base64_images(self, results: Dict, output_dir: str, is_pdf: bool = False):
        """Save base64 encoded images to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if is_pdf:
            pages = results.get('pages', [])
            for page in pages:
                if 'overlay_image' in page:
                    img_data = base64.b64decode(page['overlay_image'])
                    img_path = output_path / f"page_{page['page_number']}_overlay.png"
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    print(f"✓ Saved: {img_path}")
        else:
            if 'overlay_image' in results:
                img_data = base64.b64decode(results['overlay_image'])
                img_path = output_path / "overlay.png"
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                print(f"✓ Saved: {img_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="API Client for Document Forgery Detection"
    )
    parser.add_argument('--url', '-u', default='http://127.0.0.1:8000',
                       help='API base URL')
    parser.add_argument('--image', '-i', help='Analyze single image')
    parser.add_argument('--pdf', '-p', help='Analyze PDF')
    parser.add_argument('--batch', '-b', nargs='+', help='Batch analyze multiple files')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Check API status')
    parser.add_argument('--health', action='store_true',
                       help='Check API health')
    parser.add_argument('--save-images', action='store_true',
                       help='Save analysis images to files')
    parser.add_argument('--output', '-o', default='api_results/',
                       help='Output directory for images')
    
    args = parser.parse_args()
    
    # Initialize client
    client = ForgertyDetectionAPIClient(args.url)
    
    print(f"Document Forgery Detection API Client")
    print(f"API URL: {args.url}\n")
    
    # Health check
    if args.health:
        print("Checking API health...")
        client.health_check()
        return
    
    # Status check
    if args.status:
        print("Getting API status...")
        status = client.get_api_status()
        if status:
            print(json.dumps(status, indent=2))
        return
    
    # Analyze image
    if args.image:
        print(f"Analyzing image: {args.image}")
        results = client.analyze_image(args.image)
        client.print_results(results, args.save_images, args.output)
        return
    
    # Analyze PDF
    if args.pdf:
        print(f"Analyzing PDF: {args.pdf}")
        results = client.analyze_pdf(args.pdf)
        client.print_results(results, args.save_images, args.output)
        return
    
    # Batch analyze
    if args.batch:
        print(f"Batch analyzing {len(args.batch)} file(s)...")
        results = client.batch_analyze(args.batch)
        client.print_results(results, args.save_images, args.output)
        return
    
    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
