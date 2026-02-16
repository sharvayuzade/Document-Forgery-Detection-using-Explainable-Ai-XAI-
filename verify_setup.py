"""
Quick start guide for Document Forgery Detection System
Run this script to verify installation and test the system
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def check_python_version():
    """Check Python version"""
    print_header("1. Checking Python Version")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9:
        print_success(f"Python version OK (3.9+)")
        return True
    else:
        print_error(f"Python 3.9+ required, found {version.major}.{version.minor}")
        return False

def check_directories():
    """Check required directories"""
    print_header("2. Checking Directory Structure")
    
    project_root = Path(__file__).parent
    required_dirs = {
        'archive/CASIA2/Au': 'Authentic images',
        'archive/CASIA2/Tp': 'Tampered images',
        'src': 'Source code',
        'src/data': 'Data utilities',
        'src/models': 'Model architectures',
        'src/utils': 'Utility functions',
        'src/xai': 'XAI implementations',
        'backend': 'FastAPI backend',
        'frontend': 'Streamlit frontend',
        'models': 'Model weights directory',
        'outputs': 'Output directory'
    }
    
    all_exist = True
    for dir_path, description in required_dirs.items():
        full_path = project_root / dir_path
        if full_path.exists():
            print_success(f"{description}: {dir_path}")
        else:
            print_warning(f"{description} missing: {dir_path}")
            all_exist = False
    
    return all_exist

def check_dataset():
    """Check CASIA v2.0 dataset"""
    print_header("3. Checking CASIA v2.0 Dataset")
    
    project_root = Path(__file__).parent
    au_dir = project_root / 'archive' / 'CASIA2' / 'Au'
    tp_dir = project_root / 'archive' / 'CASIA2' / 'Tp'
    
    if au_dir.exists():
        au_images = list(au_dir.glob('**/*.[jJ][pP][gG]')) + \
                   list(au_dir.glob('**/*.[pP][nN][gG]'))
        print_success(f"Authentic images: {len(au_images)} found")
    else:
        print_error(f"Authentic images directory not found: {au_dir}")
        return False
    
    if tp_dir.exists():
        tp_images = list(tp_dir.glob('**/*.[jJ][pP][gG]')) + \
                   list(tp_dir.glob('**/*.[pP][nN][gG]'))
        print_success(f"Tampered images: {len(tp_images)} found")
    else:
        print_error(f"Tampered images directory not found: {tp_dir}")
        return False
    
    if len(au_images) > 0 and len(tp_images) > 0:
        return True
    else:
        print_error("Dataset appears to be empty")
        return False

def check_dependencies():
    """Check Python dependencies"""
    print_header("4. Checking Python Dependencies")
    
    required_packages = {
        'tensorflow': 'TensorFlow (Deep Learning)',
        'keras': 'Keras (High-level API)',
        'cv2': 'OpenCV (Computer Vision)',
        'numpy': 'NumPy (Numerical Computing)',
        'pandas': 'Pandas (Data Analysis)',
        'sklearn': 'Scikit-learn (Machine Learning)',
        'fastapi': 'FastAPI (Web Framework)',
        'streamlit': 'Streamlit (Frontend)',
        'PIL': 'Pillow (Image Processing)',
        'pdf2image': 'pdf2image (PDF Processing)',
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print_success(f"{description}: {package}")
        except ImportError:
            print_error(f"{description} not found: {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print_warning(f"\nMissing packages: {', '.join(missing_packages)}")
        print_info("Run: pip install -r requirements.txt")
        return False
    else:
        print_success("All required packages installed!")
        return True

def check_model():
    """Check if model exists"""
    print_header("5. Checking Model")
    
    project_root = Path(__file__).parent
    model_path = project_root / 'models' / 'best_model.h5'
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print_success(f"Model found: {model_path}")
        print_info(f"File size: {size_mb:.2f} MB")
        return True
    else:
        print_warning(f"Model not found: {model_path}")
        print_info("Run training: python train.py")
        return False

def check_gpu():
    """Check GPU availability"""
    print_header("6. Checking GPU Support")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print_success(f"GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print_info(f"  - {gpu}")
            return True
        else:
            print_warning("No GPU detected, will use CPU")
            print_info("For Intel i5-1220P with Iris Xe: install intel-extension-for-tensorflow")
            return False
    except Exception as e:
        print_error(f"Error checking GPU: {e}")
        return False

def print_usage_guide():
    """Print usage guide"""
    print_header("Usage Guide")
    
    print(f"""
{Colors.BOLD}Training:{Colors.ENDC}
  python train.py
  
{Colors.BOLD}Inference (Single file):{Colors.ENDC}
  python inference.py --input document.pdf
  
{Colors.BOLD}Inference (Batch):{Colors.ENDC}
  python inference.py --input images_folder --batch
  
{Colors.BOLD}FastAPI Backend:{Colors.ENDC}
  python -m backend.app
  Documentation: http://localhost:8000/docs
  
{Colors.BOLD}Streamlit Frontend:{Colors.ENDC}
  streamlit run frontend/streamlit_app.py
  Web UI: http://localhost:8501
  
{Colors.BOLD}Check Model Status:{Colors.ENDC}
  curl http://localhost:8000/api/status
  
{Colors.BOLD}Analyze Image via API:{Colors.ENDC}
  curl -X POST "http://localhost:8000/api/analyze/image" \\
    -F "file=@document.jpg"
    
{Colors.BOLD}Analyze PDF via API:{Colors.ENDC}
  curl -X POST "http://localhost:8000/api/analyze/pdf" \\
    -F "file=@document.pdf"
""")

def print_summary(results):
    """Print summary of checks"""
    print_header("Summary")
    
    checks = [
        ('Python Version', results.get('python', False)),
        ('Directory Structure', results.get('directories', False)),
        ('Dataset (CASIA v2.0)', results.get('dataset', False)),
        ('Python Dependencies', results.get('dependencies', False)),
        ('Pre-trained Model', results.get('model', False)),
        ('GPU Support', results.get('gpu', False))
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for check_name, result in checks:
        status = Colors.OKGREEN + '✓ PASS' + Colors.ENDC if result else Colors.FAIL + '✗ FAIL' + Colors.ENDC
        print(f"  {status}  {check_name}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} checks passed{Colors.ENDC}")
    
    if passed == total:
        print(Colors.OKGREEN + "\n✓ System ready for deployment!" + Colors.ENDC)
        return True
    else:
        print(Colors.WARNING + "\n⚠ Some checks failed. Please address the issues above." + Colors.ENDC)
        return False

def main():
    """Run all checks"""
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("="*80)
    print("DOCUMENT FORGERY DETECTION - SYSTEM VERIFICATION")
    print("="*80)
    print(Colors.ENDC)
    
    results = {
        'python': check_python_version(),
        'directories': check_directories(),
        'dataset': check_dataset(),
        'dependencies': check_dependencies(),
        'model': check_model(),
        'gpu': check_gpu()
    }
    
    print_usage_guide()
    success = print_summary(results)
    
    print(f"\n{Colors.BOLD}Additional Resources:{Colors.ENDC}")
    print("  📖 README.md - Complete documentation")
    print("  🚀 train.py - Start training")
    print("  🔍 inference.py - Run inference")
    print("  📊 backend/app.py - Start API")
    print("  🖥️ frontend/streamlit_app.py - Start UI")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
