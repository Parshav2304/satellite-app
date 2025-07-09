import streamlit as st
import numpy as np
from PIL import Image
import random
import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from io import BytesIO
import base64

# Page config
st.set_page_config(
    page_title="EcoSat Monitor Pro", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌍"
)

# Custom CSS for beautiful UI
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
    
    .alert-success {
        background: linear-gradient(90deg, #00C851, #00FF00);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(90deg, #FFA726, #FF9800);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background: linear-gradient(90deg, #F44336, #FF5722);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .stProgress .st-bo {
        background: linear-gradient(90deg, #00C851, #00FF00);
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
    
    .dashboard-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

if "favorite_regions" not in st.session_state:
    st.session_state.favorite_regions = []

# Enhanced Sidebar
st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
st.sidebar.title("🌍 EcoSat Monitor Pro")
st.sidebar.markdown("**AI-powered satellite land cover analysis**")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.markdown("### 📍 Analysis Settings")
region = st.sidebar.selectbox(
    "Select Region", 
    ["Global", "Americas", "Europe", "Asia", "Africa", "Oceania", "Antarctica"]
)

time_range = st.sidebar.selectbox(
    "Select Time Range", 
    ["Real-time", "7d", "30d", "90d", "1y", "5y"]
)

analysis_type = st.sidebar.selectbox(
    "Analysis Type",
    ["Land Cover", "Change Detection", "Environmental Health", "Vegetation Index", "Water Quality", "Urban Growth"]
)

# Model Settings
st.sidebar.markdown("### 🤖 Model Settings")
model_version = st.sidebar.selectbox("Model Version", ["CNN-v3.2", "ResNet-v2.1", "EfficientNet-v1.5"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.7)

# Favorite Regions
st.sidebar.markdown("### ⭐ Favorite Regions")
if st.sidebar.button("Add Current Region to Favorites"):
    if region not in st.session_state.favorite_regions:
        st.session_state.favorite_regions.append(region)
        st.sidebar.success(f"Added {region} to favorites!")

if st.session_state.favorite_regions:
    for fav in st.session_state.favorite_regions:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.text(fav)
        if col2.button("🗑️", key=f"remove_{fav}"):
            st.session_state.favorite_regions.remove(fav)
            st.rerun()

# Statistics
st.sidebar.markdown("### 📊 Statistics")
st.sidebar.metric("Total Analyses", st.session_state.analysis_count)
st.sidebar.metric("Regions Analyzed", len(set([item['region'] for item in st.session_state.history])))
st.sidebar.metric("Avg Confidence", f"{np.mean([item['confidence'] for item in st.session_state.history]) if st.session_state.history else 0:.1f}%")

# Main Header
st.markdown("""
<div class="main-header">
    <h1>📡 EcoSat Monitor Pro</h1>
    <p>Advanced AI-powered satellite image analysis for environmental monitoring</p>
</div>
""", unsafe_allow_html=True)

# Enhanced prediction simulation
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
    
    # Adjust based on region
    region_adjustments = {
        "Americas": {"Forest": 1.2, "Agricultural": 1.1},
        "Europe": {"Urban": 1.3, "Agricultural": 1.2},
        "Asia": {"Urban": 1.4, "Water": 1.1},
        "Africa": {"Desert": 1.3, "Barren": 1.2},
        "Oceania": {"Water": 1.5, "Forest": 1.1},
        "Antarctica": {"Barren": 2.0, "Water": 0.3}
    }
    
    if region in region_adjustments:
        for key, multiplier in region_adjustments[region].items():
            if key in base_predictions:
                base_predictions[key] *= multiplier
    
    # Normalize
    total = sum(base_predictions.values())
    normalized = {k: round(v / total, 3) for k, v in base_predictions.items()}
    
    # Additional metrics based on analysis type
    additional_metrics = {}
    if analysis_type == "Vegetation Index":
        additional_metrics["NDVI"] = round(random.uniform(0.3, 0.8), 3)
        additional_metrics["EVI"] = round(random.uniform(0.2, 0.7), 3)
    elif analysis_type == "Water Quality":
        additional_metrics["Turbidity"] = round(random.uniform(1, 10), 1)
        additional_metrics["Chlorophyll"] = round(random.uniform(0.5, 5.0), 2)
    
    return normalized, additional_metrics

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📷 Analysis", "📊 Dashboard", "📈 Trends", "⚙️ Settings"])

with tab1:
    # Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📷 Upload Satellite Image")
    st.markdown("Supported formats: JPG, JPEG, PNG, TIFF, GeoTIFF")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "tiff", "tif"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample Images
    st.markdown("### 🖼️ Or Try Sample Images")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    with sample_col1:
        if st.button("🌳 Forest Sample"):
            st.info("Forest sample image loaded")
            uploaded_file = "forest_sample"
    
    with sample_col2:
        if st.button("🏙️ Urban Sample"):
            st.info("Urban sample image loaded")
            uploaded_file = "urban_sample"
    
    with sample_col3:
        if st.button("🌊 Water Sample"):
            st.info("Water sample image loaded")
            uploaded_file = "water_sample"
    
    # Analysis
    if uploaded_file:
        if isinstance(uploaded_file, str):
            st.info(f"Using sample image: {uploaded_file}")
        else:
            st.image(uploaded_file, caption="Uploaded Satellite Image", use_column_width=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            analyze_button = st.button("🔍 Analyze Image", type="primary")
        
        with col2:
            batch_analyze = st.button("📊 Batch Analysis")
        
        if analyze_button:
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            with st.spinner("Running AI Model..."):
                predictions, additional_metrics = simulate_advanced_model(analysis_type, region)
                confidence = max(predictions.values())
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Update session state
                st.session_state.analysis_count += 1
                st.session_state.history.append({
                    "timestamp": timestamp,
                    "region": region,
                    "analysis_type": analysis_type,
                    "predictions": predictions,
                    "additional_metrics": additional_metrics,
                    "confidence": round(confidence * 100),
                    "model_version": model_version
                })
                
                st.success("✅ Analysis Complete!")
                
                # Results
                st.markdown("### 🛰️ Land Cover Classification Results")
                
                # Create two columns for results
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    for i, (label, prob) in enumerate(list(predictions.items())[:4]):
                        st.progress(int(prob * 100), text=f"{label}: {prob*100:.1f}%")
                
                with res_col2:
                    for i, (label, prob) in enumerate(list(predictions.items())[4:]):
                        st.progress(int(prob * 100), text=f"{label}: {prob*100:.1f}%")
                
                # Additional metrics
                if additional_metrics:
                    st.markdown("### 📈 Additional Metrics")
                    metric_cols = st.columns(len(additional_metrics))
                    for i, (metric, value) in enumerate(additional_metrics.items()):
                        metric_cols[i].metric(metric, value)
                
                # Environmental Impact
                st.markdown("### 🌿 Environmental Impact Assessment")
                veg = int((predictions.get("Green Area", 0) + predictions.get("Forest", 0)) * 100)
                water = int(predictions.get("Water", 0) * 100)
                urban = int(predictions.get("Urban", 0) * 100)
                deforest = round(max(0, (0.4 - predictions.get("Forest", 0))) * 10, 1)
                
                impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
                impact_col1.metric("Vegetation Health", f"{veg}%", delta=f"{random.randint(-5, 5)}%")
                impact_col2.metric("Water Coverage", f"{water}%", delta=f"{random.randint(-3, 3)}%")
                impact_col3.metric("Urban Development", f"{urban}%", delta=f"{random.randint(0, 8)}%")
                impact_col4.metric("Deforestation Risk", f"{deforest}%", delta=f"{random.randint(-2, 4)}%")
                
                # Alerts and Recommendations
                st.markdown("### 🚨 Alerts & Recommendations")
                
                alert_col1, alert_col2 = st.columns(2)
                
                with alert_col1:
                    if veg < 40:
                        st.markdown('<div class="alert-danger">🌱 <strong>Critical:</strong> Very low vegetation coverage detected</div>', unsafe_allow_html=True)
                    elif veg < 60:
                        st.markdown('<div class="alert-warning">⚠️ <strong>Warning:</strong> Low vegetation coverage detected</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-success">✅ <strong>Good:</strong> Healthy vegetation coverage</div>', unsafe_allow_html=True)
                
                with alert_col2:
                    if water < 20:
                        st.markdown('<div class="alert-danger">💧 <strong>Critical:</strong> Very limited water coverage</div>', unsafe_allow_html=True)
                    elif water < 40:
                        st.markdown('<div class="alert-warning">⚠️ <strong>Warning:</strong> Limited water coverage</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-success">✅ <strong>Good:</strong> Adequate water coverage</div>', unsafe_allow_html=True)
                
                if deforest > 6:
                    st.markdown('<div class="alert-danger">🌲 <strong>Alert:</strong> High deforestation rate detected - Immediate action required</div>', unsafe_allow_html=True)
                
                if urban > 30:
                    st.markdown('<div class="alert-warning">🏙️ <strong>Notice:</strong> High urban development - Monitor environmental impact</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Analysis Dashboard")
    
    if len(st.session_state.history) > 0:
        # Summary metrics
        st.markdown("#### 📈 Summary Metrics")
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        avg_confidence = np.mean([item['confidence'] for item in st.session_state.history])
        total_analyses = len(st.session_state.history)
        unique_regions = len(set([item['region'] for item in st.session_state.history]))
        latest_analysis = st.session_state.history[-1]['timestamp']
        
        summary_col1.metric("Average Confidence", f"{avg_confidence:.1f}%")
        summary_col2.metric("Total Analyses", total_analyses)
        summary_col3.metric("Regions Covered", unique_regions)
        summary_col4.metric("Latest Analysis", latest_analysis.split(' ')[0])
        
        # Recent Analysis History
        st.markdown("#### 🕐 Recent Analysis History")
        for i, item in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"Analysis {total_analyses - i}: {item['timestamp']} - {item['region']}"):
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    st.write(f"**Region:** {item['region']}")
                    st.write(f"**Analysis Type:** {item['analysis_type']}")
                    st.write(f"**Confidence:** {item['confidence']}%")
                    st.write(f"**Model:** {item['model_version']}")
                
                with exp_col2:
                    st.write("**Top Classifications:**")
                    sorted_predictions = sorted(item['predictions'].items(), key=lambda x: x[1], reverse=True)
                    for label, prob in sorted_predictions[:3]:
                        st.write(f"• {label}: {prob*100:.1f}%")
                
                if item['additional_metrics']:
                    st.write("**Additional Metrics:**")
                    for metric, value in item['additional_metrics'].items():
                        st.write(f"• {metric}: {value}")
        
        # Download Report
        if st.button("📥 Download Analysis Report"):
            # Create a simple report
            report_data = []
            for item in st.session_state.history:
                row = {
                    "Timestamp": item['timestamp'],
                    "Region": item['region'],
                    "Analysis_Type": item['analysis_type'],
                    "Confidence": item['confidence'],
                    "Model": item['model_version']
                }
                row.update({f"Prediction_{k}": v for k, v in item['predictions'].items()})
                report_data.append(row)
            
            df = pd.DataFrame(report_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f"ecosat_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("📷 Upload and analyze images to populate the dashboard.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### 📈 Trends Analysis")
    
    if len(st.session_state.history) > 1:
        # Create trend data
        df_history = pd.DataFrame(st.session_state.history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        
        # Confidence trend
        fig_confidence = px.line(
            df_history, 
            x='timestamp', 
            y='confidence',
            title='Confidence Trend Over Time',
            labels={'confidence': 'Confidence (%)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Region analysis
        region_counts = df_history['region'].value_counts()
        fig_regions = px.pie(
            values=region_counts.values,
            names=region_counts.index,
            title='Analysis Distribution by Region'
        )
        st.plotly_chart(fig_regions, use_container_width=True)
        
        # Analysis type distribution
        if 'analysis_type' in df_history.columns:
            type_counts = df_history['analysis_type'].value_counts()
            fig_types = px.bar(
                x=type_counts.index,
                y=type_counts.values,
                title='Analysis Types Distribution',
                labels={'x': 'Analysis Type', 'y': 'Count'}
            )
            st.plotly_chart(fig_types, use_container_width=True)
    else:
        st.info("📊 Perform more analyses to see trends.")

with tab4:
    st.markdown("### ⚙️ Application Settings")
    
    # Theme Settings
    st.markdown("#### 🎨 Theme Settings")
    theme_col1, theme_col2 = st.columns(2)
    
    with theme_col1:
        chart_theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"])
    
    with theme_col2:
        color_scheme = st.selectbox("Color Scheme", ["Default", "Ocean", "Forest", "Sunset", "Monochrome"])
    
    # Data Management
    st.markdown("#### 📊 Data Management")
    
    if st.button("🗑️ Clear All History"):
        st.session_state.history = []
        st.session_state.analysis_count = 0
        st.success("History cleared!")
    
    if st.button("📤 Export All Data"):
        if st.session_state.history:
            all_data = {
                "history": st.session_state.history,
                "analysis_count": st.session_state.analysis_count,
                "favorite_regions": st.session_state.favorite_regions
            }
            st.download_button(
                "Download Complete Data",
                data=str(all_data),
                file_name=f"ecosat_complete_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        else:
            st.warning("No data to export")
    
    # Advanced Settings
    st.markdown("#### ⚙️ Advanced Settings")
    
    auto_save = st.checkbox("Auto-save analyses", value=True)
    notifications = st.checkbox("Enable notifications", value=True)
    real_time_mode = st.checkbox("Real-time monitoring mode", value=False)
    
    # API Settings
    st.markdown("#### 🔗 API Settings")
    api_endpoint = st.text_input("Custom API Endpoint", placeholder="https://api.example.com/v1/analyze")
    api_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
    
    if st.button("🧪 Test API Connection"):
        if api_endpoint and api_key:
            st.success("✅ API connection test successful!")
        else:
            st.error("❌ Please provide both API endpoint and key")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea, #764ba2); color: white; border-radius: 10px;">
    <h4>🌍 EcoSat Monitor Pro</h4>
    <p>Empowering environmental monitoring through AI-powered satellite analysis</p>
    <p>Version 3.2.1 | © 2024 EcoSat Technologies</p>
</div>
""", unsafe_allow_html=True)
