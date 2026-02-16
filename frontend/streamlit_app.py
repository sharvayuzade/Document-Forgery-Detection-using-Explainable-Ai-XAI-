"""
Streamlit frontend for document forgery detection with Grad-CAM visualization
"""

import streamlit as st
import numpy as np
import cv2
import logging
from pathlib import Path
import tempfile
import sys
import requests
from PIL import Image
import base64
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import *
from data.data_loader import PDFProcessor
from utils.preprocessing import ImagePreprocessor
from xai.gradcam import GradCAM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Document Forgery Detector",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .authentic-badge {
        color: #00AA00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .tampered-badge {
        color: #FF0000;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Global state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0


# Helper functions
@st.cache_resource
def load_model_and_components():
    """Load trained model and Grad-CAM"""
    try:
        import tensorflow as tf
        
        logger.info("Loading model...")
        if BEST_MODEL_PATH.exists():
            model = tf.keras.models.load_model(str(BEST_MODEL_PATH))
        else:
            st.error(f"Model not found at {BEST_MODEL_PATH}")
            st.stop()
        
        # Find appropriate layer for Grad-CAM
        layer_name = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
        
        if layer_name is None:
            layer_name = model.layers[-3].name
        
        try:
            gradcam = GradCAM(model, layer_name)
        except:
            gradcam = None
        
        preprocessor = ImagePreprocessor(target_size=(INPUT_SIZE, INPUT_SIZE))
        pdf_processor = PDFProcessor(dpi=PDF_DPI)
        
        return model, gradcam, preprocessor, pdf_processor
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None


def process_image(image: np.ndarray, model, gradcam, preprocessor):
    """Process single image with model and Grad-CAM"""
    try:
        # Preprocess
        preprocessed = preprocessor.preprocess_image(image, apply_ela=False, normalize=True)
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        # Predict
        prediction = model.predict(preprocessed, verbose=0)[0][0]
        
        # Grad-CAM
        if gradcam:
            heatmap = gradcam.compute_gradcam(preprocessed)[0]
            overlay = gradcam.overlay_heatmap(image, heatmap, alpha=HEATMAP_ALPHA)
        else:
            heatmap = None
            overlay = None
        
        return prediction, heatmap, overlay
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None, None, None


def display_metrics(prediction: float, heatmap: np.ndarray):
    """Display metrics in columns"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Prediction Score",
            f"{prediction:.4f}",
            delta=None,
            delta_color="off"
        )
    
    with col2:
        confidence = prediction if prediction > 0.5 else 1.0 - prediction
        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%",
            delta=None,
            delta_color="off"
        )
    
    with col3:
        if heatmap is not None:
            forgery_score = np.mean(heatmap)
            st.metric(
                "Forgery Score",
                f"{forgery_score:.4f}",
                delta=None,
                delta_color="off"
            )


def display_result_cards(prediction: float, heatmap: np.ndarray):
    """Display result cards"""
    col1, col2 = st.columns(2)
    
    is_tampered = prediction > CONFIDENCE_THRESHOLD
    status = "🔴 TAMPERED" if is_tampered else "✅ AUTHENTIC"
    confidence_color = "#FF4444" if is_tampered else "#44FF44"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Document Status</h3>
            <p style="color: {confidence_color}; font-size: 2rem;">{status}</p>
            <p>Prediction: {prediction:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if heatmap is not None:
            suspicious_pixels = np.sum(heatmap > 0.5)
            total_pixels = heatmap.shape[0] * heatmap.shape[1]
            suspicious_percentage = 100 * suspicious_pixels / total_pixels
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>Heatmap Analysis</h3>
                <p>Suspicious Pixels: {suspicious_pixels:,} / {total_pixels:,}</p>
                <p>Coverage: {suspicious_percentage:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)


# Main app
def main():
    # Header
    st.markdown("""
    <h1 style="text-align: center; color: #1f77b4;">
        🔐 Document Forgery Detection System
    </h1>
    <p style="text-align: center; color: #666;">
        Powered by ResNet50, Grad-CAM, and Error Level Analysis
    </p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Load models
    model, gradcam, preprocessor, pdf_processor = load_model_and_components()
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Detection Threshold")
        threshold = st.slider(
            "Confidence threshold for tampered classification",
            0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05
        )
        
        st.subheader("Analysis Options")
        apply_ela = st.checkbox("Apply Error Level Analysis", value=True)
        show_heatmap = st.checkbox("Show Grad-CAM Heatmap", value=True)
        show_overlay = st.checkbox("Show Overlay", value=True)
        heatmap_alpha = st.slider("Heatmap Transparency", 0.0, 1.0, HEATMAP_ALPHA, 0.1)
        
        st.divider()
        st.subheader("📊 Model Info")
        st.info("""
        **Architecture**: ResNet50 with Transfer Learning
        **Input Size**: 224×224 pixels
        **Classes**: Authentic, Tampered
        **XAI Method**: Grad-CAM
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "📊 Analysis", "📚 Batch"])
    
    # Tab 1: Single image upload
    with tab1:
        st.subheader("Upload Document")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload an image or PDF",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'],
                help="Supported formats: JPG, PNG, BMP, TIFF, PDF"
            )
        
        with col2:
            process_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"File: {uploaded_file.name}")
            with col2:
                st.info(f"Size: {uploaded_file.size / 1024:.2f} KB")
            with col3:
                st.info(f"Type: {uploaded_file.type}")
            
            if process_button:
                with st.spinner("🔄 Processing..."):
                    try:
                        # Save temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                            tmp.write(uploaded_file.getbuffer())
                            tmp_path = tmp.name
                        
                        # Process based on file type
                        if uploaded_file.type == 'application/pdf' or uploaded_file.name.endswith('.pdf'):
                            # PDF processing
                            images = pdf_processor.pdf_to_images(tmp_path)
                            
                            if len(images) > 0:
                                st.session_state.analysis_results = []
                                
                                for page_num, img in enumerate(images):
                                    pred, hmap, overlay = process_image(img, model, gradcam, preprocessor)
                                    
                                    st.session_state.analysis_results.append({
                                        'page': page_num + 1,
                                        'image': img,
                                        'prediction': pred,
                                        'heatmap': hmap,
                                        'overlay': overlay
                                    })
                                
                                st.success(f"✅ Analyzed {len(images)} page(s)")
                            else:
                                st.error("Could not extract images from PDF")
                        else:
                            # Image processing
                            image = cv2.imread(tmp_path)
                            if image is not None:
                                pred, hmap, overlay = process_image(image, model, gradcam, preprocessor)
                                
                                st.session_state.analysis_results = [{
                                    'page': 1,
                                    'image': image,
                                    'prediction': pred,
                                    'heatmap': hmap,
                                    'overlay': overlay
                                }]
                                
                                st.success("✅ Analysis complete")
                            else:
                                st.error("Could not read image file")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
                        logger.error(f"Error: {e}", exc_info=True)
    
    # Tab 2: Analysis results
    with tab2:
        if st.session_state.analysis_results is not None and len(st.session_state.analysis_results) > 0:
            st.subheader("📊 Analysis Results")
            
            results = st.session_state.analysis_results
            
            # Navigation for multi-page documents
            if len(results) > 1:
                col1, col2, col3 = st.columns([1, 4, 1])
                with col1:
                    if st.button("⬅️ Previous"):
                        st.session_state.current_page = max(0, st.session_state.current_page - 1)
                with col2:
                    page_num = st.selectbox(
                        "Select page",
                        range(len(results)),
                        format_func=lambda x: f"Page {results[x]['page']}",
                        index=st.session_state.current_page
                    )
                    st.session_state.current_page = page_num
                with col3:
                    if st.button("Next ➡️"):
                        st.session_state.current_page = min(len(results) - 1, st.session_state.current_page + 1)
            else:
                page_num = 0
            
            result = results[page_num]
            
            # Display metrics
            st.markdown("### 📈 Metrics")
            display_metrics(result['prediction'], result['heatmap'])
            
            st.divider()
            
            # Display result cards
            display_result_cards(result['prediction'], result['heatmap'])
            
            st.divider()
            
            # Display visualizations
            st.markdown("### 🖼️ Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image_rgb = cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB)
                st.image(image_rgb, use_column_width=True)
            
            with col2:
                if show_overlay and result['overlay'] is not None:
                    st.subheader("Grad-CAM Overlay")
                    overlay_rgb = cv2.cvtColor(result['overlay'], cv2.COLOR_BGR2RGB)
                    st.image(overlay_rgb, use_column_width=True)
                elif show_heatmap and result['heatmap'] is not None:
                    st.subheader("Heatmap")
                    st.image(result['heatmap'], use_column_width=True, clim=(0, 1))
            
            if show_heatmap and result['heatmap'] is not None:
                st.markdown("### 🌡️ Heatmap Details")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Max Intensity", f"{np.max(result['heatmap']):.4f}")
                with col2:
                    st.metric("Mean Intensity", f"{np.mean(result['heatmap']):.4f}")
                with col3:
                    st.metric("Min Intensity", f"{np.min(result['heatmap']):.4f}")
                
                # Display heatmap as image
                heatmap_uint8 = (result['heatmap'] * 255).astype(np.uint8)
                st.image(heatmap_uint8, use_column_width=True, clim=(0, 255))
        
        else:
            st.info("📤 Upload a document in the 'Upload' tab to see analysis results")
    
    # Tab 3: Batch processing
    with tab3:
        st.subheader("📚 Batch Analysis")
        
        uploaded_files = st.file_uploader(
            "Upload multiple files",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'pdf'],
            accept_multiple_files=True,
            help="Upload multiple images or PDFs for batch analysis"
        )
        
        if uploaded_files and st.button("🔍 Analyze All", type="primary"):
            with st.spinner("Processing batch..."):
                batch_results = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    try:
                        # Save temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                            tmp.write(file.getbuffer())
                            tmp_path = tmp.name
                        
                        # Process
                        if file.name.endswith('.pdf'):
                            images = pdf_processor.pdf_to_images(tmp_path)
                            for page_num, img in enumerate(images):
                                pred, _, _ = process_image(img, model, gradcam, preprocessor)
                                batch_results.append({
                                    'file': file.name,
                                    'page': page_num + 1,
                                    'prediction': pred,
                                    'status': '🔴 TAMPERED' if pred > threshold else '✅ AUTHENTIC'
                                })
                        else:
                            image = cv2.imread(tmp_path)
                            if image is not None:
                                pred, _, _ = process_image(image, model, gradcam, preprocessor)
                                batch_results.append({
                                    'file': file.name,
                                    'page': 1,
                                    'prediction': pred,
                                    'status': '🔴 TAMPERED' if pred > threshold else '✅ AUTHENTIC'
                                })
                    
                    except Exception as e:
                        logger.error(f"Error processing {file.name}: {e}")
                        batch_results.append({
                            'file': file.name,
                            'error': str(e)
                        })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Display results
                st.success(f"✅ Batch analysis complete ({len(batch_results)} items)")
                
                # Create dataframe
                import pandas as pd
                df = pd.DataFrame(batch_results)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        'file': 'File Name',
                        'page': 'Page',
                        'prediction': st.column_config.NumberColumn('Prediction', format='%.4f'),
                        'status': 'Status'
                    }
                )
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyzed", len(batch_results))
                with col2:
                    authentic = sum(1 for r in batch_results if '✅' in r.get('status', ''))
                    st.metric("Authentic", authentic)
                with col3:
                    tampered = sum(1 for r in batch_results if '🔴' in r.get('status', ''))
                    st.metric("Tampered", tampered)


if __name__ == "__main__":
    main()
