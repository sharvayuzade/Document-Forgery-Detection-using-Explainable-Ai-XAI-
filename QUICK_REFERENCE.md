# Quick Reference Guide - Document Forgery Detection System

## 🚀 30-Second Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python verify_setup.py

# 3. Train model
python train.py

# 4. Start API (Terminal 1)
python -m backend.app

# 5. Start UI (Terminal 2)
streamlit run frontend/streamlit_app.py

# 6. Open browser
# UI: http://localhost:8501
# API docs: http://localhost:8000/docs
```

---

## 📋 Common Commands

### Training

```bash
# Start training with default settings
python train.py

# Expected output:
# ✓ Loaded 5000+ authentic images
# ✓ Loaded 5000+ tampered images
# ✓ Model trained successfully
# ✓ Test accuracy: ~95%+
```

### Inference

```bash
# Analyze single image
python inference.py --input document.jpg

# Analyze PDF (multi-page)
python inference.py --input document.pdf

# Batch analyze directory
python inference.py --input images_folder --batch

# With custom output dir
python inference.py --input document.pdf --output results/
```

### API Usage

```bash
# Check API status
curl http://127.0.0.1:8000/api/health

# Analyze image
curl -X POST "http://127.0.0.1:8000/api/analyze/image" \
  -F "file=@document.jpg"

# Analyze PDF
curl -X POST "http://127.0.0.1:8000/api/analyze/pdf" \
  -F "file=@document.pdf"

# Batch analyze
curl -X POST "http://127.0.0.1:8000/api/batch/analyze" \
  -F "files=@doc1.jpg" \
  -F "files=@doc2.pdf"
```

### Python Client

```bash
# Check API health
python api_client.py --health

# Analyze image
python api_client.py --image document.jpg --save-images --output results/

# Analyze PDF
python api_client.py --pdf document.pdf --save-images

# Batch analyze
python api_client.py --batch doc1.jpg doc2.pdf doc3.png --save-images

# Get API status
python api_client.py --status
```

---

## 📂 Important Directories

| Path | Purpose |
|------|---------|
| `src/` | Source code (models, data, utils, XAI) |
| `archive/CASIA2/` | Dataset (Au, Tp, Groundtruth) |
| `models/` | Trained model weights |
| `outputs/` | Generated heatmaps & predictions |
| `logs/` | Training & inference logs |
| `backend/` | FastAPI application |
| `frontend/` | Streamlit UI |

---

## 🎯 Output Interpretation

### Prediction Score
- **0.0 - 0.5**: Authentic (Green ✓)
- **0.5 - 1.0**: Tampered (Red ✗)

### Confidence
- **0 - 25%**: Very uncertain
- **25 - 50%**: Somewhat uncertain
- **50 - 75%**: Confident
- **75 - 100%**: Very confident

### Forgery Score
- **0.0 - 0.1**: No tampering
- **0.1 - 0.3**: Minor tampering
- **0.3 - 0.5**: Moderate tampering
- **0.5 - 1.0**: Severe tampering

### Heatmap Colors
- 🔵 **Blue**: Authentic (no tampering)
- 🟢 **Green**: Moderate concern
- 🟡 **Yellow**: High concern
- 🔴 **Red**: Strong tampering indication

---

## ⚙️ Configuration Quick Reference

Edit `src/config.py`:

```python
# Model
INPUT_SIZE = 224                    # Image size
BATCH_SIZE = 16                     # Smaller if out of memory
EPOCHS = 30                         # Training iterations

# Data
TRAIN_SPLIT = 0.7                 # 70% train
VAL_SPLIT = 0.15                  # 15% validation
TEST_SPLIT = 0.15                 # 15% test

# Hardware
USE_MIXED_PRECISION = True         # Reduces memory usage
MAX_WORKERS = 4                    # CPU threads

# Detection
CONFIDENCE_THRESHOLD = 0.5          # Classification threshold
HEATMAP_ALPHA = 0.4                # Overlay transparency

# PDF
PDF_DPI = 150                       # Conversion resolution
PDF_MAX_PAGES = 50                  # Max pages per PDF
```

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Memory error | Reduce BATCH_SIZE in config.py |
| Slow training | Reduce TRAIN_SPLIT, use smaller images |
| API won't start | Check port 8000 is free |
| PDF not working | Install Python Poppler: `pip install python-poppler-qt5` |
| GPU not detected | Install Intel GPU drivers & `intel-extension-for-tensorflow` |
| Model not found | Run `python train.py` first |

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~95-98% |
| Precision | ~94-97% |
| Recall | ~93-96% |
| Inference Time | ~100-150ms per image |

---

## 🔗 API Endpoints

### GET Endpoints

```
GET /api/health              - Check API health
GET /api/status              - Get API configuration
```

### POST Endpoints

```
POST /api/analyze/image      - Analyze single image
POST /api/analyze/pdf        - Analyze PDF document
POST /api/batch/analyze      - Batch analyze multiple files
```

---

## 📖 Documentation Files

- **README.md** - Complete setup & usage guide
- **GRADCAM_DOCUMENTATION.md** - Technical details & heatmap interpretation
- **GRADCAM_DOCUMENTATION.pdf** (auto-generated) - Printable version

---

## 💡 Example Workflows

### Workflow 1: Train & Infer

```bash
# Step 1: Train model
python train.py

# Step 2: Single image inference
python inference.py --input test_image.jpg --output results/

# Step 3: Check results
# Open results/reports/analysis_*.json
```

### Workflow 2: API Server

```bash
# Terminal 1: Start API
python -m backend.app

# Terminal 2: Test API
python api_client.py --image test.jpg --save-images

# Terminal 3: Check results
# Open outputs/overlays/
# Open outputs/reports/
```

### Workflow 3: Full Pipeline

```bash
# Terminal 1: API
python -m backend.app

# Terminal 2: Streamlit UI
streamlit run frontend/streamlit_app.py

# Browser:
# 1. Go to http://localhost:8501
# 2. Upload image/PDF
# 3. View predictions & heatmaps
# 4. Download results
```

---

## 🔐 Security Notes (Production)

For production deployment:

1. **Enable HTTPS** in FastAPI
2. **Set SECRET_KEY** in `.env`
3. **Configure CORS** properly
4. **Rate limit** API endpoints
5. **Add authentication** (JWT tokens)
6. **Validate** file types & sizes
7. **Sandbox** file processing
8. **Monitor** API usage & errors

---

## 📱 System Requirements

**Minimum:**
- Intel i5 (7th gen+)
- 8 GB RAM
- 2 GB GPU memory
- 10 GB disk space

**Recommended:**
- Intel i5-1220P or better
- 16 GB RAM
- 4 GB GPU memory
- 20 GB disk space
- SSD storage

---

## 🎓 Learning Resources

1. **Grad-CAM Paper**: arxiv.org/abs/1610.02055
2. **ResNet Paper**: arxiv.org/abs/1512.03385
3. **Document Forgery**: NIST DFSLW dataset
4. **TensorFlow Guide**: tensorflow.org/guide
5. **FastAPI Docs**: fastapi.tiangolo.com

---

## 📞 Getting Help

1. **Check Logs**: `logs/training.log` or `logs/inference.log`
2. **Run Health Check**: `python verify_setup.py`
3. **Test API**: `curl http://localhost:8000/api/health`
4. **Review Code**: Check docstrings in source files
5. **Documentation**: Read GRADCAM_DOCUMENTATION.md

---

## ✅ Pre-deployment Checklist

- [ ] All dependencies installed
- [ ] CASIA v2.0 dataset extracted
- [ ] Model trained & saved
- [ ] API tested with sample files
- [ ] UI tested in browser
- [ ] Output directory structure verified
- [ ] Logs reviewed for errors
- [ ] Documentation copied to server
- [ ] Configuration hardened for production
- [ ] Security measures implemented

---

**Last Updated**: February 16, 2026  
**Version**: 1.0.0  
**System**: Intel i5-1220P + Intel Iris Xe
