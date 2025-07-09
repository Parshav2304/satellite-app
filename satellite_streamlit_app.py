import streamlit as st
import numpy as np
import pandas as pd
import random
import datetime
from PIL import Image
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go

# 🔧 Page Config
st.set_page_config(
    page_title="Satellite Image Classifier", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍"
)

# 🎨 Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #228B22, #32CD32);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        color: white;
    }
    .analysis-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 🧠 Session State Initialization
if "history" not in st.session_state:
    st.session_state.history = []

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

# 🧭 Sidebar Layout
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.title("Satellite Image Classifier")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("### 📊 Statistics")
st.sidebar.metric("Total Analyses", st.session_state.analysis_count)
st.sidebar.metric(
    "Regions Analyzed", 
    len(set([item['region'] for item in st.session_state.history]))
)
avg_conf = (
    np.mean([item['confidence'] for item in st.session_state.history]) 
    if st.session_state.history else 0
)
st.sidebar.metric("Avg Confidence", f"{avg_conf:.1f}%")

# 🖼️ Header
st.markdown("""
<div class="main-header">
    <h1>Satellite Image Classifier</h1>
    <p>AI-powered predictions from space data 🚀</p>
</div>
""", unsafe_allow_html=True)

# 🔬 Simulate prediction engine
def simulate_advanced_model(analysis_type, region):
    base_predictions = {
        "Green Area": random.uniform(0.2, 0.6),
        "Water": random.uniform(0.1, 0.4),
        "Desert": random.uniform(0.1, 0.3),
        "Urban": random.uniform(0.05, 0.25),
        "Cloudy": random.uniform(0.05, 0.3),
        "Agricultural": random.uniform(0.1, 0.4),
        "Forest": random.uniform(0.1, 0.5),
        "Barren": random.uniform(0.05, 0.2)
    }
    
    region_adjustments = {
        "Americas": {"Forest": 1.2, "Agricultural": 1.1},
        "Europe": {"Urban": 1.3, "Agricultural": 1.2},
        "Asia": {"Urban": 1.4, "Water": 1.1},
        "Africa": {"Desert": 1.3, "Barren": 1.2},
        "Oceania": {"Water": 1.5, "Forest": 1.1},
        "Antarctica": {"Barren": 2.0, "Water": 0.3}
    }

    # Apply region-based tweaks
    for k, v in region_adjustments.get(region, {}).items():
        base_predictions[k] *= v

    total = sum(base_predictions.values())
    normalized = {k: round(v / total, 3) for k, v in base_predictions.items()}

    metrics = {}
    if analysis_type == "Vegetation Index":
        metrics["NDVI"] = round(random.uniform(0.3, 0.8), 3)
        metrics["EVI"] = round(random.uniform(0.2, 0.7), 3)
    elif analysis_type == "Water Quality":
        metrics["Turbidity"] = round(random.uniform(1, 10), 1)
        metrics["Chlorophyll"] = round(random.uniform(0.5, 5.0), 2)

    return normalized, metrics

# 📷 Analysis Tab
tab1, = st.tabs(["📷 Analysis"])

with tab1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### Upload Satellite Image")
    st.markdown("Supported formats: JPG, JPEG, PNG")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

    analysis_type = st.selectbox("Choose Analysis Type", ["Vegetation Index", "Water Quality"])
    region = st.selectbox("Select Region", ["Americas", "Europe", "Asia", "Africa", "Oceania", "Antarctica"])
    model_version = "3.2.1"

    if uploaded_file:
        st.image(uploaded_file, caption="Satellite Image Preview", use_column_width=True)
        if st.button("🔍 Analyze Image"):
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            with st.spinner("Processing with eco-intelligence..."):
                predictions, metrics = simulate_advanced_model(analysis_type, region)
                confidence = max(predictions.values())
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Log result
                st.session_state.analysis_count += 1
                st.session_state.history.append({
                    "timestamp": timestamp,
                    "region": region,
                    "analysis_type": analysis_type,
                    "predictions": predictions,
                    "additional_metrics": metrics,
                    "confidence": round(confidence * 100),
                    "model_version": model_version
                })

                st.success("✅ Analysis Complete!")

                # 📊 Prediction Bars
                st.markdown("### Results")
                col1, col2 = st.columns(2)
                pred_items = list(predictions.items())

                for i in range(len(pred_items)):
                    col = col1 if i < 4 else col2
                    label, prob = pred_items[i]
                    col.progress(int(prob * 100), text=f"{label}: {prob*100:.1f}%")

                # 📈 Additional Metrics
                if metrics:
                    st.markdown("### 📈 Environmental Indicators")
                    cols = st.columns(len(metrics))
                    for i, (name, value) in enumerate(metrics.items()):
                        cols[i].metric(name, value)

            st.markdown('</div>', unsafe_allow_html=True)

# 🧾 Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea, #764ba2); color: white; border-radius: 10px;">
    <h4>Satellite Image Classifier</h4>
    <p>Version 2.1.2 | © 2025 PKS TECH</p>
</div>
""", unsafe_allow_html=True)
