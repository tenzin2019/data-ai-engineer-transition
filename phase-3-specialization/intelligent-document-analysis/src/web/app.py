"""
Streamlit web interface for the Intelligent Document Analysis System.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.document_processor import DocumentProcessor
from core.ai_analyzer import AIAnalyzer
from utils.file_utils import validate_file_type, get_file_extension, is_streamlit_file_size_valid_for_standard, is_streamlit_file_size_valid_for_large, get_streamlit_upload_type_for_file
from config.settings import settings
from services.document_service import init_database, save_document_to_db, get_documents_from_db, get_document_by_id, delete_document_from_db, clear_all_documents_from_db

# Page configuration
st.set_page_config(
    page_title="Intelligent Document Analysis System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with Portfolio-Consistent Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables - Light Theme */
    :root {
        --primary-50: #f0f9ff;
        --primary-100: #e0f2fe;
        --primary-200: #bae6fd;
        --primary-300: #7dd3fc;
        --primary-400: #38bdf8;
        --primary-500: #0ea5e9;
        --primary-600: #0284c7;
        --primary-700: #0369a1;
        --primary-800: #075985;
        --primary-900: #0c4a6e;
        --secondary-50: #f8fafc;
        --secondary-100: #f1f5f9;
        --secondary-200: #e2e8f0;
        --secondary-300: #cbd5e1;
        --secondary-400: #94a3b8;
        --secondary-500: #64748b;
        --secondary-600: #475569;
        --secondary-700: #334155;
        --secondary-800: #1e293b;
        --secondary-900: #0f172a;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        
        /* Theme-aware colors */
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --border-primary: #e2e8f0;
        --border-secondary: #cbd5e1;
    }
    
    /* Dark Theme Variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-50: #0c4a6e;
            --primary-100: #075985;
            --primary-200: #0369a1;
            --primary-300: #0284c7;
            --primary-400: #0ea5e9;
            --primary-500: #38bdf8;
            --primary-600: #7dd3fc;
            --primary-700: #bae6fd;
            --primary-800: #e0f2fe;
            --primary-900: #f0f9ff;
            --secondary-50: #0f172a;
            --secondary-100: #1e293b;
            --secondary-200: #334155;
            --secondary-300: #475569;
            --secondary-400: #64748b;
            --secondary-500: #94a3b8;
            --secondary-600: #cbd5e1;
            --secondary-700: #e2e8f0;
            --secondary-800: #f1f5f9;
            --secondary-900: #f8fafc;
            --accent-purple: #a78bfa;
            --accent-cyan: #22d3ee;
            --success: #34d399;
            --warning: #fbbf24;
            --error: #f87171;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.3);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.3);
            
            /* Dark theme-aware colors */
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-primary: #334155;
            --border-secondary: #475569;
        }
    }
    
    /* Global Styles */
    * {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 50%, var(--accent-cyan) 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        animation: gradient-shift 3s ease infinite;
        letter-spacing: -0.02em;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Modern Card Design */
    .modern-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-primary);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-lg);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        color: var(--text-primary);
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-xl);
        border-color: var(--primary-300);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Analysis Sections */
    .analysis-section {
        background: var(--bg-secondary);
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-sm);
        transition: all 0.3s ease;
        border: 1px solid var(--border-primary);
        color: var(--text-primary);
    }
    
    .analysis-section:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: var(--primary-300);
        background: var(--bg-tertiary);
    }
    
    /* Entity Tags */
    .entity-tag {
        background: linear-gradient(135deg, var(--primary-100) 0%, var(--primary-200) 100%);
        color: var(--primary-700);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        display: inline-block;
        border: 1px solid var(--primary-200);
        transition: all 0.2s ease;
    }
    
    .entity-tag:hover {
        background: linear-gradient(135deg, var(--primary-200) 0%, var(--primary-300) 100%);
        transform: translateY(-1px);
        box-shadow: var(--shadow-sm);
    }
    
    /* Sentiment Styling */
    .sentiment-positive {
        color: var(--success);
        font-weight: 600;
    }
    
    .sentiment-negative {
        color: var(--error);
        font-weight: 600;
    }
    
    .sentiment-neutral {
        color: var(--warning);
        font-weight: 600;
    }
    
    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid var(--border-primary);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.3s ease;
        background: transparent;
        color: var(--text-secondary);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--primary-50);
        color: var(--primary-600);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-500) 0%, var(--accent-purple) 100%);
        color: white;
        box-shadow: var(--shadow-md);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        padding-top: 2rem;
        background: var(--bg-secondary);
        color: var(--text-primary);
    }
    
    .css-1d391kg .stSelectbox,
    .css-1d391kg .stSlider,
    .css-1d391kg .stCheckbox {
        margin-bottom: 1rem;
    }
    
    /* Enhanced File Uploader */
    .stFileUploader {
        border: 3px dashed var(--primary-300);
        border-radius: 16px;
        padding: 3rem 2rem;
        background: var(--bg-secondary);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stFileUploader::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, var(--primary-100) 50%, transparent 70%);
        transform: translateX(-100%);
        transition: transform 0.6s;
    }
    
    .stFileUploader:hover::before {
        transform: translateX(100%);
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-500);
        background: var(--bg-tertiary);
        transform: translateY(-4px);
        box-shadow: var(--shadow-lg);
    }
    
    .stFileUploader > div {
        text-align: center;
        position: relative;
        z-index: 1;
    }
    
    .stFileUploader > div > div {
        color: var(--primary-600);
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Upload Instructions */
    .upload-instructions {
        text-align: center;
        color: var(--primary-600);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        padding: 2rem;
        background: var(--bg-secondary);
        border-radius: 16px;
        border: 3px dashed var(--primary-300);
        position: relative;
        overflow: hidden;
    }
    
    .upload-instructions::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, var(--primary-100), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-500) 0%, var(--accent-purple) 100%);
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-md);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--primary-700) 0%, var(--accent-purple) 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Alert Styling */
    .stAlert {
        margin: 1rem 0;
        border-radius: 12px;
        border: none;
        box-shadow: var(--shadow-sm);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border-left: 4px solid var(--success);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
        border-left: 4px solid var(--error);
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(14, 165, 233, 0.05) 100%);
        border-left: 4px solid var(--primary-500);
    }
    
    /* Form Elements */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid var(--border-primary);
        transition: all 0.3s ease;
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .stSelectbox > div > div:focus,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-500);
        box-shadow: 0 0 0 3px var(--primary-100);
    }
    
    /* Metrics Container */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-item {
        background: var(--bg-primary);
        border: 1px solid var(--border-primary);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        color: var(--text-primary);
    }
    
    .metric-item:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-color: var(--primary-300);
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(14, 165, 233, 0.3);
        border-radius: 50%;
        border-top-color: var(--primary-500);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-container {
            grid-template-columns: 1fr;
        }
        
        .modern-card {
            padding: 1rem;
        }
    }
    
    /* Additional theme-aware styling */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .stApp > div {
        background: var(--bg-primary);
    }
    
    /* Streamlit component overrides for theme compatibility */
    .stMarkdown {
        color: var(--text-primary);
    }
    
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6 {
        color: var(--text-primary);
    }
    
    .stMarkdown p {
        color: var(--text-secondary);
    }
    
    /* Ensure proper contrast in both themes */
    .stSelectbox label,
    .stSlider label,
    .stCheckbox label {
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)

# Initialize database only if not disabled
if os.getenv("DB_DISABLED", "false").lower() != "true":
    init_database()
else:
    print("✅ Database disabled, skipping initialization")

# Initialize session state
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "ready"

# Initialize components
@st.cache_resource
def get_document_processor():
    return DocumentProcessor()

@st.cache_resource
def get_ai_analyzer():
    return AIAnalyzer()

def main():
    """Main application function."""
    
    # Modern header with enhanced styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem; padding: 2rem 0; background: linear-gradient(135deg, var(--primary-50) 0%, var(--secondary-50) 100%); border-radius: 20px; position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: linear-gradient(45deg, transparent 30%, rgba(14, 165, 233, 0.05) 50%, transparent 70%); animation: shimmer 3s infinite;"></div>
        <h1 class="main-header" style="position: relative; z-index: 1; margin-bottom: 1rem;">📄 Intelligent Document Analysis System</h1>
        <p style="font-size: 1.2rem; color: var(--secondary-600); font-weight: 500; margin: 0; position: relative; z-index: 1;">
            Transform your documents with AI-powered insights and analysis
        </p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; position: relative; z-index: 1;">
            <div style="
                background: linear-gradient(135deg, var(--primary-100) 0%, var(--primary-200) 100%);
                color: var(--primary-700);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
                border: 1px solid var(--primary-200);
            ">
                🤖 AI-Powered Analysis
            </div>
            <div style="
                background: linear-gradient(135deg, var(--accent-purple) 0%, rgba(139, 92, 246, 0.2) 100%);
                color: var(--accent-purple);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
                border: 1px solid rgba(139, 92, 246, 0.3);
            ">
                📊 Real-time Insights
            </div>
            <div style="
                background: linear-gradient(135deg, var(--accent-cyan) 0%, rgba(6, 182, 212, 0.2) 100%);
                color: var(--accent-cyan);
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: 500;
                border: 1px solid rgba(6, 182, 212, 0.3);
            ">
                ⚡ Fast Processing
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern sidebar
    with st.sidebar:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            text-align: center;
        ">
            <h3 style="margin: 0; font-size: 1.3rem; font-weight: 700;">🔧 Configuration</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; opacity: 0.9;">Customize your analysis settings</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Document type selection
        st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h4 style="color: var(--primary-700); margin-bottom: 0.5rem; font-size: 1.1rem;">📄 Document Type</h4>
        </div>
        """, unsafe_allow_html=True)
        
        document_type = st.selectbox(
            "Select document type for context-specific analysis",
            ["general", "legal", "financial", "technical", "medical", "business"],
            help="Choose the document type to get more accurate analysis results",
            key="main_document_type"
        )
        
        # Analysis options with modern styling
        st.markdown("""
        <div style="margin: 2rem 0 1.5rem 0;">
            <h4 style="color: var(--primary-700); margin-bottom: 1rem; font-size: 1.1rem;">🎯 Analysis Options</h4>
            <div style="
                background: linear-gradient(135deg, var(--primary-50) 0%, var(--secondary-50) 100%);
                padding: 1rem;
                border-radius: 12px;
                border: 1px solid var(--primary-200);
            ">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            include_entities = st.checkbox("🏷️ Entities", value=True, key="main_entities", help="Extract named entities from the document")
            include_sentiment = st.checkbox("😊 Sentiment", value=True, key="main_sentiment", help="Analyze document sentiment")
        with col2:
            include_summary = st.checkbox("📝 Summary", value=True, key="main_summary", help="Generate document summary")
            include_recommendations = st.checkbox("💡 Recommendations", value=True, key="main_recommendations", help="Generate actionable recommendations")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced options with modern styling
        st.markdown("""
        <div style="margin: 2rem 0 1.5rem 0;">
            <h4 style="color: var(--primary-700); margin-bottom: 1rem; font-size: 1.1rem;">⚙️ Advanced Settings</h4>
            <div style="
                background: linear-gradient(135deg, var(--secondary-50) 0%, var(--primary-50) 100%);
                padding: 1rem;
                border-radius: 12px;
                border: 1px solid var(--secondary-200);
            ">
        """, unsafe_allow_html=True)
        
        with st.expander("🔧 AI Model Parameters", expanded=False):
            max_tokens = st.slider("Max Tokens", 1000, 8000, 4000, key="main_max_tokens", help="Maximum tokens for AI processing")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, key="main_temperature", help="Controls randomness in AI responses")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add a modern footer to sidebar
        st.markdown("""
        <div style="
            margin-top: 2rem;
            padding: 1rem;
            background: linear-gradient(135deg, var(--secondary-100) 0%, var(--primary-100) 100%);
            border-radius: 12px;
            text-align: center;
            border: 1px solid var(--secondary-200);
        ">
            <p style="margin: 0; color: var(--secondary-600); font-size: 0.9rem;">
                💡 <strong>Pro Tip:</strong> Choose the right document type for better analysis accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload & Analyze", "📊 Analysis Results", "📈 Analytics Dashboard", "⚙️ Settings", "🤖 Model Comparison"])
    
    with tab1:
        upload_and_analyze_tab(document_type, include_entities, include_sentiment, include_summary, include_recommendations)
    
    with tab2:
        analysis_results_tab()
    
    with tab3:
        analytics_dashboard_tab()
    
    with tab4:
        settings_tab()
    
    with tab5:
        from model_comparison import show_model_comparison, show_smart_selection_demo
        show_model_comparison()
        st.markdown("---")
        show_smart_selection_demo()

def upload_and_analyze_tab(document_type, include_entities, include_sentiment, include_summary, include_recommendations):
    """Upload and analyze documents tab."""
    
    # Modern header with gradient text
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        ">📤 Upload & Analyze</h2>
        <p style="color: var(--secondary-600); font-size: 1.1rem; margin: 0;">
            Transform your documents with AI-powered insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern upload interface with enhanced styling
    st.markdown("""
    <div class="upload-instructions">
        <div style="font-size: 4rem; margin-bottom: 1.5rem; animation: float 3s ease-in-out infinite;">📁</div>
        <h3 style="color: var(--primary-600); margin-bottom: 1rem; font-size: 1.5rem; font-weight: 600;">
            Drag and drop your document here
        </h3>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1.5rem; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: 600; color: var(--primary-700); margin-bottom: 0.5rem;">📄 Supported Formats</div>
                <div style="color: var(--secondary-600);">PDF, DOCX, XLSX, TXT</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.2rem; font-weight: 600; color: var(--primary-700); margin-bottom: 0.5rem;">📏 File Size Limits</div>
                <div style="color: var(--secondary-600);">Up to 15MB (standard)<br>or 200MB (large files)</div>
            </div>
        </div>
        <div style="
            background: linear-gradient(135deg, var(--primary-100) 0%, var(--secondary-100) 100%);
            padding: 1rem;
            border-radius: 12px;
            color: var(--primary-600);
            font-weight: 500;
            display: inline-block;
        ">
            💡 Smart size detection automatically optimizes your upload experience
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Single file uploader with smart size detection
    uploaded_file = st.file_uploader(
        "Choose a document file",
        type=['pdf', 'docx', 'xlsx', 'txt'],
        help="📄 Supported formats: PDF, DOCX, XLSX, TXT\n📏 Automatic size detection: 15MB (standard) or 200MB (large)\n💡 Drag and drop files directly onto the area above",
        label_visibility="collapsed",
        key="unified_upload"
    )
    
    # Determine upload type based on file size
    upload_type = None
    if uploaded_file is not None:
        upload_type = get_streamlit_upload_type_for_file(uploaded_file)
    
    if uploaded_file is not None:
        # Validate file based on upload type
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        is_valid_type = validate_file_type(uploaded_file.name, uploaded_file.type)
        
        # Check size validation based on upload type
        if upload_type == "standard":
            is_valid_size = is_streamlit_file_size_valid_for_standard(uploaded_file)
            max_size = settings.max_file_size_standard / (1024 * 1024)
        elif upload_type == "large":
            is_valid_size = is_streamlit_file_size_valid_for_large(uploaded_file)
            max_size = settings.max_file_size_large / (1024 * 1024)
        else:  # too_large
            is_valid_size = False
            max_size = settings.max_file_size_large / (1024 * 1024)
        
        is_valid = is_valid_type and is_valid_size
        
        # Display file information in modern card format
        st.markdown('<div class="modern-card">', unsafe_allow_html=True)
        st.markdown("### 📋 File Information")
        
        # Create a grid layout for file info
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-item">
                <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">File Name</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700); word-break: break-all;">
                    {uploaded_file.name[:25] + "..." if len(uploaded_file.name) > 25 else uploaded_file.name}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-item">
                <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Size</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                    {file_size_mb:.1f} MB
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            file_extension = get_file_extension(uploaded_file.name)
            st.markdown(f"""
            <div class="metric-item">
                <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Type</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                    {file_extension.upper() if file_extension else "Unknown"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-item">
                <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Upload Type</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                    {upload_type.title()}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            status_icon = "✅" if is_valid else "❌"
            status_color = "var(--success)" if is_valid else "var(--error)"
            st.markdown(f"""
            <div class="metric-item">
                <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Status</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: {status_color};">
                    {status_icon} {"Valid" if is_valid else "Invalid"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show validation messages
        if not is_valid_type:
            st.error("❌ Invalid file type. Please upload PDF, DOCX, XLSX, or TXT files.")
        elif not is_valid_size:
            if upload_type == "too_large":
                st.error(f"❌ File too large. Size: {file_size_mb:.1f}MB exceeds maximum limit of {max_size:.0f}MB. Please compress your file or split it into smaller parts.")
            else:
                st.error(f"❌ File validation failed. Size: {file_size_mb:.1f}MB, Expected limit: {max_size:.0f}MB.")
        else:
            st.success(f"✅ File validated successfully for {upload_type} upload ({file_size_mb:.1f}MB).")
        
        # Modern analyze button with enhanced styling
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <style>
                .analyze-button {
                    background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
                    color: white;
                    border: none;
                    border-radius: 16px;
                    padding: 1rem 3rem;
                    font-size: 1.2rem;
                    font-weight: 700;
                    cursor: pointer;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    box-shadow: var(--shadow-lg);
                    position: relative;
                    overflow: hidden;
                }
                
                .analyze-button:hover {
                    transform: translateY(-3px);
                    box-shadow: var(--shadow-xl);
                    background: linear-gradient(135deg, var(--primary-700) 0%, var(--accent-purple) 100%);
                }
                
                .analyze-button:active {
                    transform: translateY(-1px);
                }
                
                .analyze-button:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                    transform: none;
                }
                
                .analyze-button::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
                    transition: left 0.5s;
                }
                
                .analyze-button:hover::before {
                    left: 100%;
                }
            </style>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze Document", type="primary", disabled=not is_valid, key="analyze_button"):
                # Get advanced options from session state
                max_tokens = st.session_state.get('main_max_tokens', 4000)
                temperature = st.session_state.get('main_temperature', 0.3)
                analyze_document(uploaded_file, document_type, include_entities, include_sentiment, include_summary, include_recommendations, max_tokens, temperature, upload_type)

def analyze_document(uploaded_file, document_type, include_entities, include_sentiment, include_summary, include_recommendations, max_tokens=4000, temperature=0.3, upload_type="standard"):
    """Analyze uploaded document."""
    
    st.session_state.processing_status = "processing"
    
    # Create modern progress container
    st.markdown("""
    <div class="modern-card" style="text-align: center; margin: 2rem 0;">
        <h3 style="color: var(--primary-600); margin-bottom: 1.5rem; font-size: 1.5rem;">
            🤖 AI Analysis in Progress
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create progress bar with custom styling
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Add loading animation
    st.markdown("""
    <div style="text-align: center; margin: 1rem 0;">
        <div class="loading-spinner" style="margin: 0 auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Step 1: Save uploaded file
        status_text.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, var(--primary-50) 0%, var(--secondary-50) 100%); border-radius: 12px; margin: 1rem 0;">
            <div style="font-size: 1.1rem; color: var(--primary-600); font-weight: 500;">
                📁 Saving uploaded file...
            </div>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(10)
        
        # Create temporary file
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Step 2: Process document
        status_text.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, var(--primary-50) 0%, var(--secondary-50) 100%); border-radius: 12px; margin: 1rem 0;">
            <div style="font-size: 1.1rem; color: var(--primary-600); font-weight: 500;">
                📄 Extracting text from document...
            </div>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(30)
        
        document_processor = get_document_processor()
        processing_result = document_processor.process_document(temp_file_path, uploaded_file.type)
        
        # Step 3: AI Analysis
        status_text.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, var(--accent-purple) 0%, var(--primary-50) 100%); border-radius: 12px; margin: 1rem 0;">
            <div style="font-size: 1.1rem; color: var(--primary-600); font-weight: 500;">
                🤖 Running AI analysis...
            </div>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(60)
        
        ai_analyzer = get_ai_analyzer()
        analysis_result = ai_analyzer.analyze_document(
            processing_result['text'], 
            document_type,
            max_tokens,
            temperature,
            include_entities,
            include_sentiment,
            include_summary,
            include_recommendations
        )
        
        # Step 4: Combine results
        status_text.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, var(--accent-cyan) 0%, var(--primary-50) 100%); border-radius: 12px; margin: 1rem 0;">
            <div style="font-size: 1.1rem; color: var(--primary-600); font-weight: 500;">
                📊 Processing results...
            </div>
        </div>
        """, unsafe_allow_html=True)
        progress_bar.progress(90)
        
        # Save to database or session state
        if os.getenv("DB_DISABLED", "false").lower() == "true":
            # Database disabled - store in session state
            document_record = {
                'id': 1,  # Dummy ID
                'filename': uploaded_file.name,
                'original_filename': uploaded_file.name,
                'file_size': len(uploaded_file.getvalue()),
                'mime_type': uploaded_file.type,
                'document_type': document_type,
                'text_length': len(processing_result['text']),
                'page_count': processing_result.get('page_count', 1),
                'upload_time': datetime.now().isoformat(),
                'analysis_result': analysis_result,
                'analysis_options': {
                    'include_entities': include_entities,
                    'include_sentiment': include_sentiment,
                    'include_summary': include_summary,
                    'include_recommendations': include_recommendations
                }
            }
            st.session_state.current_analysis = document_record
            print("✅ Document stored in session state (database disabled)")
        else:
            # Database enabled - save to database
            document_id = save_document_to_db(
                filename=uploaded_file.name,
                original_filename=uploaded_file.name,
                file_path=str(temp_file_path),
                file_size=len(uploaded_file.getvalue()),
                mime_type=uploaded_file.type,
                document_type=document_type,
                extracted_text=processing_result['text'],
                text_length=len(processing_result['text']),
                page_count=processing_result.get('page_count', 1),
                summary=analysis_result.get('summary', ''),
                key_phrases=analysis_result.get('key_phrases', []),
                sentiment_score=analysis_result.get('sentiment', {}).get('score'),
                confidence_score=analysis_result.get('confidence_score'),
                analysis_data=analysis_result
            )
            
            # Get the saved document for display
            document_record = get_document_by_id(document_id)
            st.session_state.current_analysis = document_record
        
        # Step 5: Complete
        progress_bar.progress(100)
        status_text.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, var(--success) 0%, var(--primary-50) 100%); border-radius: 16px; margin: 1rem 0; box-shadow: var(--shadow-lg);">
            <div style="font-size: 1.3rem; color: white; font-weight: 700; margin-bottom: 0.5rem;">
                ✅ Analysis Complete!
            </div>
            <div style="font-size: 1rem; color: rgba(255, 255, 255, 0.9);">
                Your document has been successfully analyzed
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean up temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()
        
        st.session_state.processing_status = "completed"
        
        # Show modern success message
        st.markdown("""
        <div class="modern-card" style="text-align: center; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%); border-left: 4px solid var(--success);">
            <h3 style="color: var(--success); margin-bottom: 1rem; font-size: 1.5rem;">
                🎉 Document Analysis Completed Successfully!
            </h3>
            <p style="color: var(--secondary-600); font-size: 1.1rem; margin: 0;">
                Your document has been processed and analyzed. Check the "Analysis Results" tab to view detailed insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add celebration animation
        st.balloons()
        
    except Exception as e:
        st.error(f"Error analyzing document: {str(e)}")
        st.session_state.processing_status = "error"
        
        # Clean up temporary file
        if 'temp_file_path' in locals() and temp_file_path.exists():
            temp_file_path.unlink()

def analysis_results_tab():
    """Display analysis results tab."""
    
    # Modern header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        ">📊 Analysis Results</h2>
        <p style="color: var(--secondary-600); font-size: 1.1rem; margin: 0;">
            Explore detailed insights from your analyzed documents
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get documents from database or session state
    if os.getenv("DB_DISABLED", "false").lower() == "true":
        # Database disabled - use session state
        documents = [st.session_state.current_analysis] if st.session_state.current_analysis else []
    else:
        # Database enabled - get from database
        documents = get_documents_from_db()
    
    if not documents:
        st.markdown("""
        <div class="modern-card" style="text-align: center; background: linear-gradient(135deg, var(--primary-50) 0%, var(--secondary-50) 100%);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
            <h3 style="color: var(--primary-600); margin-bottom: 1rem; font-size: 1.5rem;">
                No Documents Analyzed Yet
            </h3>
            <p style="color: var(--secondary-600); font-size: 1.1rem; margin: 0;">
                Upload and analyze a document first to see detailed results here.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Document selection with modern styling
    st.markdown("""
    <div class="modern-card" style="margin-bottom: 2rem;">
        <h3 style="color: var(--primary-600); margin-bottom: 1rem; font-size: 1.3rem;">
            📋 Select Document to View
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    document_options = {f"{doc['filename']} ({doc['upload_time'].strftime('%Y-%m-%d %H:%M')})": doc for doc in documents}
    selected_doc_name = st.selectbox("Select Document", list(document_options.keys()), key="document_selector")
    selected_doc = document_options[selected_doc_name]
    
    # Display document information in modern card format
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Document Overview")
    
    # Create a grid layout for document info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">File Name</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700); word-break: break-all;">
                {selected_doc['filename'][:30] + "..." if len(selected_doc['filename']) > 30 else selected_doc['filename']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">File Size</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {selected_doc['file_size_mb']:.2f} MB
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Text Length</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {selected_doc['text_length']:,} characters
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Pages</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {selected_doc['page_count']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis results
    analysis = selected_doc['analysis_result']
    analysis_options = selected_doc.get('analysis_options', {})
    
    # Summary section
    if analysis_options.get('include_summary', True) and analysis.get('summary'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("📝 Summary")
        st.write(analysis['summary'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Key phrases
    if analysis_options.get('include_entities', True) and analysis.get('key_phrases'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("🔑 Key Phrases")
        phrases = analysis['key_phrases'][:10]  # Show top 10
        for phrase in phrases:
            st.markdown(f'<span class="entity-tag">{phrase}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Entities
    if analysis_options.get('include_entities', True) and analysis.get('entities'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("🏷️ Extracted Entities")
        
        # Group entities by type
        entity_types = {}
        for entity in analysis['entities']:
            entity_type = entity.get('type', 'OTHER')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
        
        # Display entities by type
        for entity_type, entities in entity_types.items():
            st.write(f"**{entity_type}:**")
            for entity in entities[:5]:  # Show top 5 per type
                confidence = entity.get('confidence', 0)
                st.write(f"• {entity['text']} (confidence: {confidence:.2f})")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sentiment analysis
    if analysis_options.get('include_sentiment', True) and analysis.get('sentiment'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("😊 Sentiment Analysis")
        
        sentiment = analysis['sentiment']
        score = sentiment.get('score', 0)
        label = sentiment.get('label', 'neutral')
        
        # Determine sentiment class
        if label == 'positive':
            sentiment_class = 'sentiment-positive'
        elif label == 'negative':
            sentiment_class = 'sentiment-negative'
        else:
            sentiment_class = 'sentiment-neutral'
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", label.title())
        with col2:
            st.metric("Score", f"{score:.2f}")
        
        # Sentiment gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Sentiment Score"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightgray"},
                    {'range': [-0.3, 0.3], 'color': "yellow"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Topics
    if analysis_options.get('include_entities', True) and analysis.get('topics'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("📚 Topics")
        topics = analysis['topics']
        for topic in topics:
            st.write(f"• {topic}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    if analysis_options.get('include_recommendations', True) and analysis.get('recommendations'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("💡 Recommendations")
        recommendations = analysis['recommendations']
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Raw data (expandable)
    with st.expander("🔍 Raw Analysis Data"):
        st.json(analysis)

def analytics_dashboard_tab():
    """Analytics dashboard tab."""
    
    # Modern header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        ">📈 Analytics Dashboard</h2>
        <p style="color: var(--secondary-600); font-size: 1.1rem; margin: 0;">
            Comprehensive insights and analytics from your document analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get documents from database or session state
    if os.getenv("DB_DISABLED", "false").lower() == "true":
        # Database disabled - use session state
        documents = [st.session_state.current_analysis] if st.session_state.current_analysis else []
    else:
        # Database enabled - get from database
        documents = get_documents_from_db()
    
    if not documents:
        st.markdown("""
        <div class="modern-card" style="text-align: center; background: linear-gradient(135deg, var(--primary-50) 0%, var(--secondary-50) 100%);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: var(--primary-600); margin-bottom: 1rem; font-size: 1.5rem;">
                No Analytics Data Available
            </h3>
            <p style="color: var(--secondary-600); font-size: 1.1rem; margin: 0;">
                Upload and analyze some documents first to see analytics and insights.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Overall statistics with modern styling
    total_docs = len(documents)
    total_pages = sum(doc['page_count'] or 0 for doc in documents)
    total_text_length = sum(doc['text_length'] or 0 for doc in documents)
    avg_confidence = sum(doc['analysis_result'].get('confidence_score', 0) or 0 for doc in documents) / total_docs if total_docs > 0 else 0
    
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Overall Statistics")
    
    # Create a grid layout for statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{total_docs}</div>
            <div style="font-size: 1rem; font-weight: 500;">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{total_pages}</div>
            <div style="font-size: 1rem; font-weight: 500;">Pages</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{total_text_length:,}</div>
            <div style="font-size: 1rem; font-weight: 500;">Characters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">{avg_confidence:.2f}</div>
            <div style="font-size: 1rem; font-weight: 500;">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document types distribution
    st.markdown("**📊 Document Types**")
    doc_types = [doc['document_type'] for doc in documents]
    type_counts = pd.Series(doc_types).value_counts()
    
    fig = px.pie(values=type_counts.values, names=type_counts.index, title="Document Types")
    st.plotly_chart(fig, use_container_width=True)
    
    # File sizes distribution
    st.markdown("**📏 File Sizes**")
    file_sizes = [doc['file_size_mb'] or 0 for doc in documents]
    
    fig = go.Figure(data=[go.Histogram(x=file_sizes, nbinsx=10)])
    fig.update_layout(title="File Sizes (MB)", height=300)
    fig.update_xaxes(title="File Size (MB)")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment analysis over time
    st.markdown("**😊 Sentiment Trends**")
    sentiments = []
    dates = []
    
    for doc in documents:
        sentiment = doc['analysis_result'].get('sentiment', {})
        if sentiment:
            sentiments.append(sentiment.get('score', 0))
            dates.append(doc['upload_time'])
    
    if sentiments:
        df_sentiment = pd.DataFrame({
            'Date': dates,
            'Sentiment Score': sentiments
        })
        
        fig = px.line(df_sentiment, x='Date', y='Sentiment Score', title="Sentiment Trends", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Entity types distribution
    st.markdown("**🏷️ Entity Types**")
    all_entities = []
    for doc in documents:
        entities = doc['analysis_result'].get('entities', [])
        for entity in entities:
            all_entities.append(entity.get('type', 'OTHER'))
    
    if all_entities:
        entity_counts = pd.Series(all_entities).value_counts()
        fig = go.Figure(data=[go.Bar(x=entity_counts.index, y=entity_counts.values)])
        fig.update_layout(title="Entity Types")
        fig.update_xaxes(title="Entity Type")
        fig.update_yaxes(title="Count")
        st.plotly_chart(fig, use_container_width=True)

def settings_tab():
    """Settings and configuration tab."""
    
    # Modern header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        ">⚙️ Settings & Configuration</h2>
        <p style="color: var(--secondary-600); font-size: 1.1rem; margin: 0;">
            Configure your system settings and preferences
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Azure OpenAI Configuration
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 🔧 Azure OpenAI Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Azure OpenAI Endpoint", value=settings.azure_openai_endpoint or "", disabled=True, key="settings_endpoint")
        st.text_input("API Version", value=settings.azure_openai_api_version, disabled=True, key="settings_api_version")
    
    with col2:
        st.text_input("Deployment Name", value=settings.azure_openai_deployment_name, disabled=True, key="settings_deployment")
        st.slider("Max Tokens", value=settings.max_tokens, min_value=1000, max_value=8000, disabled=True, key="settings_max_tokens")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File Upload Settings
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 📁 File Upload Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Max File Size (Standard)</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {settings.max_file_size_standard / (1024*1024):.0f} MB
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Max File Size (Large)</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {settings.max_file_size_large / (1024*1024):.0f} MB
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Max Document Length</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {settings.max_document_length:,} characters
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, var(--primary-50) 0%, var(--secondary-50) 100%); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--primary-200);">
            <h4 style="color: var(--primary-700); margin-bottom: 1rem; font-size: 1.1rem;">📄 Allowed File Types</h4>
        """, unsafe_allow_html=True)
        for file_type in settings.allowed_file_types:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, var(--primary-100) 0%, var(--primary-200) 100%);
                color: var(--primary-700);
                padding: 0.5rem 1rem;
                border-radius: 8px;
                margin: 0.5rem 0;
                font-weight: 500;
                border: 1px solid var(--primary-200);
            ">
                • {file_type}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # System Information
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">App Version</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {settings.app_version}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Environment</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {settings.environment.title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        debug_status = "Enabled" if settings.debug else "Disabled"
        debug_color = "var(--warning)" if settings.debug else "var(--success)"
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">Debug Mode</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {debug_color};">
                {debug_status}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-item">
            <div style="font-size: 0.875rem; color: var(--secondary-600); margin-bottom: 0.5rem; font-weight: 500;">API Port</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: var(--primary-700);">
                {settings.api_port}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear data button
    st.markdown('<div class="modern-card">', unsafe_allow_html=True)
    st.markdown("### 🗑️ Data Management")
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2); margin: 1rem 0;">
        <p style="color: var(--error); font-weight: 500; margin-bottom: 1rem;">
            ⚠️ This action will permanently delete all analysis data from the database.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if there are documents to clear
    documents = get_documents_from_db()
    doc_count = len(documents) if documents else 0
    
    if doc_count == 0:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2); margin: 1rem 0;">
            <div style="color: var(--primary-600); font-weight: 500; font-size: 1.1rem;">
                ℹ️ No analysis data found in database to clear.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.2); margin: 1rem 0;">
            <div style="color: var(--warning); font-weight: 500; font-size: 1.1rem;">
                📊 Found {doc_count} document(s) in database. Click the button below to clear all data.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ Clear All Analysis Data", type="secondary", key="clear_data_button", disabled=(doc_count == 0)):
            with st.spinner("Clearing database..."):
                if clear_all_documents_from_db():
                    st.session_state.current_analysis = None
                    st.markdown("""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%); border-radius: 12px; border: 1px solid var(--success); margin: 1rem 0;">
                        <div style="color: var(--success); font-weight: 600; font-size: 1.1rem;">
                            ✅ All analysis data has been cleared from database!
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()  # Refresh the page to update the UI
                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%); border-radius: 12px; border: 1px solid var(--error); margin: 1rem 0;">
                        <div style="color: var(--error); font-weight: 600; font-size: 1.1rem;">
                            ❌ Failed to clear data from database. Please check the console for error details.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
