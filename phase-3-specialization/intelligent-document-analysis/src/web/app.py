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

# Page configuration
st.set_page_config(
    page_title="Intelligent Document Analysis System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.75rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.25rem 0;
    }
    
    .analysis-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .entity-tag {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.4rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.1rem;
        display: inline-block;
    }
    
    .sentiment-positive {
        color: #4caf50;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #f44336;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #ff9800;
        font-weight: bold;
    }
    
    /* Compact spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
    
    /* Reduce default margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Compact sidebar */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Compact metrics */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .metric-item {
        flex: 1;
        min-width: 120px;
    }
    
    /* Compact success/error messages */
    .stAlert {
        margin: 0.5rem 0;
    }
    
    /* Compact form elements */
    .stSelectbox, .stTextInput, .stTextArea {
        margin-bottom: 0.5rem;
    }
    
    /* Enhanced file uploader styling */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e8ff 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    .stFileUploader > div {
        text-align: center;
    }
    
    .stFileUploader > div > div {
        color: #667eea;
        font-weight: 500;
    }
    
    /* Upload area text styling */
    .upload-instructions {
        text-align: center;
        color: #667eea;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        border-radius: 8px;
        border: 2px dashed #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
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
    
    # Header
    st.markdown('<h1 class="main-header">📄 Intelligent Document Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Configuration")
        
        # Document type selection
        document_type = st.selectbox(
            "Document Type",
            ["general", "legal", "financial", "technical", "medical", "business"],
            help="Select the type of document for context-specific analysis",
            key="main_document_type"
        )
        
        # Analysis options in columns
        st.markdown("**Analysis Options**")
        col1, col2 = st.columns(2)
        with col1:
            include_entities = st.checkbox("Entities", value=True, key="main_entities")
            include_sentiment = st.checkbox("Sentiment", value=True, key="main_sentiment")
        with col2:
            include_summary = st.checkbox("Summary", value=True, key="main_summary")
            include_recommendations = st.checkbox("Recommendations", value=True, key="main_recommendations")
        
        # Advanced options
        with st.expander("⚙️ Advanced", expanded=False):
            max_tokens = st.slider("Max Tokens", 1000, 8000, 4000, key="main_max_tokens")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, key="main_temperature")
    
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
    
    st.markdown("### 📤 Upload Document")
    
    # Single unified upload interface
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9ff 0%, #e8f0ff 100%);
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
        <h3 style="color: #667eea; margin-bottom: 0.5rem;">Drag and drop your document here</h3>
        <p style="color: #666; margin-bottom: 1rem;">
            <strong>Supported formats:</strong> PDF, DOCX, XLSX, TXT<br>
            <strong>File size limits:</strong> Up to 15MB (standard) or 200MB (large files)
        </p>
        <p style="color: #999; font-size: 0.9rem;">
            💡 The system will automatically determine the appropriate upload type based on your file size
        </p>
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
        
        # Display file information in compact format
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("File Name", uploaded_file.name[:20] + "..." if len(uploaded_file.name) > 20 else uploaded_file.name)
        
        with col2:
            st.metric("Size", f"{file_size_mb:.1f} MB")
        
        with col3:
            file_extension = get_file_extension(uploaded_file.name)
            st.metric("Type", file_extension.upper() if file_extension else "Unknown")
        
        with col4:
            st.metric("Upload Type", upload_type.title())
        
        with col5:
            if is_valid:
                st.metric("Status", "✅ Valid")
            else:
                st.metric("Status", "❌ Invalid")
        
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
        
        # Analyze button
        if st.button("🔍 Analyze Document", type="primary", disabled=not is_valid, key="analyze_button"):
            # Get advanced options from session state
            max_tokens = st.session_state.get('main_max_tokens', 4000)
            temperature = st.session_state.get('main_temperature', 0.3)
            analyze_document(uploaded_file, document_type, include_entities, include_sentiment, include_summary, include_recommendations, max_tokens, temperature, upload_type)

def analyze_document(uploaded_file, document_type, include_entities, include_sentiment, include_summary, include_recommendations, max_tokens=4000, temperature=0.3, upload_type="standard"):
    """Analyze uploaded document."""
    
    st.session_state.processing_status = "processing"
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Save uploaded file
        status_text.text("📁 Saving uploaded file...")
        progress_bar.progress(10)
        
        # Create temporary file
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file_path = temp_dir / uploaded_file.name
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Step 2: Process document
        status_text.text("📄 Extracting text from document...")
        progress_bar.progress(30)
        
        document_processor = get_document_processor()
        processing_result = document_processor.process_document(temp_file_path, uploaded_file.type)
        
        # Step 3: AI Analysis
        status_text.text("🤖 Running AI analysis...")
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
        status_text.text("📊 Processing results...")
        progress_bar.progress(90)
        
        # Create document record
        document_record = {
            'id': len(st.session_state.documents) + 1,
            'filename': uploaded_file.name,
            'file_type': get_file_extension(uploaded_file.name),
            'file_size_mb': len(uploaded_file.getvalue()) / (1024 * 1024),
            'upload_time': datetime.now(),
            'document_type': document_type,
            'upload_type': upload_type,
            'processing_result': processing_result,
            'analysis_result': analysis_result,
            'text_length': len(processing_result['text']),
            'page_count': processing_result.get('page_count', 1),
            'analysis_options': {
                'include_entities': include_entities,
                'include_sentiment': include_sentiment,
                'include_summary': include_summary,
                'include_recommendations': include_recommendations,
                'max_tokens': max_tokens,
                'temperature': temperature
            }
        }
        
        # Add to session state
        st.session_state.documents.append(document_record)
        st.session_state.current_analysis = document_record
        
        # Step 5: Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Clean up temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()
        
        st.session_state.processing_status = "completed"
        
        # Show success message
        st.success("Document analysis completed successfully!")
        st.balloons()
        
    except Exception as e:
        st.error(f"Error analyzing document: {str(e)}")
        st.session_state.processing_status = "error"
        
        # Clean up temporary file
        if 'temp_file_path' in locals() and temp_file_path.exists():
            temp_file_path.unlink()

def analysis_results_tab():
    """Display analysis results tab."""
    
    st.header("📊 Analysis Results")
    
    if not st.session_state.documents:
        st.info("No documents have been analyzed yet. Please upload and analyze a document first.")
        return
    
    # Document selection
    document_options = {f"{doc['filename']} ({doc['upload_time'].strftime('%Y-%m-%d %H:%M')})": doc for doc in st.session_state.documents}
    selected_doc_name = st.selectbox("Select Document", list(document_options.keys()), key="document_selector")
    selected_doc = document_options[selected_doc_name]
    
    # Display document information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("File Name", selected_doc['filename'])
    
    with col2:
        st.metric("File Size", f"{selected_doc['file_size_mb']:.2f} MB")
    
    with col3:
        st.metric("Text Length", f"{selected_doc['text_length']:,} characters")
    
    with col4:
        st.metric("Pages", selected_doc['page_count'])
    
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
    
    st.markdown("### 📈 Analytics Dashboard")
    
    if not st.session_state.documents:
        st.info("No documents analyzed yet. Upload and analyze some documents first.")
        return
    
    # Overall statistics
    total_docs = len(st.session_state.documents)
    total_pages = sum(doc['page_count'] for doc in st.session_state.documents)
    total_text_length = sum(doc['text_length'] for doc in st.session_state.documents)
    avg_confidence = sum(doc['analysis_result'].get('confidence_score', 0) for doc in st.session_state.documents) / total_docs
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", total_docs)
    
    with col2:
        st.metric("Pages", total_pages)
    
    with col3:
        st.metric("Text Length", f"{total_text_length:,}")
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Document types distribution
    st.markdown("**📊 Document Types**")
    doc_types = [doc['document_type'] for doc in st.session_state.documents]
    type_counts = pd.Series(doc_types).value_counts()
    
    fig = px.pie(values=type_counts.values, names=type_counts.index, title="Document Types")
    st.plotly_chart(fig, use_container_width=True)
    
    # File sizes distribution
    st.markdown("**📏 File Sizes**")
    file_sizes = [doc['file_size_mb'] for doc in st.session_state.documents]
    
    fig = go.Figure(data=[go.Histogram(x=file_sizes, nbinsx=10)])
    fig.update_layout(title="File Sizes (MB)", height=300)
    fig.update_xaxes(title="File Size (MB)")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment analysis over time
    st.markdown("**😊 Sentiment Trends**")
    sentiments = []
    dates = []
    
    for doc in st.session_state.documents:
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
    for doc in st.session_state.documents:
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
    
    st.header("⚙️ Settings & Configuration")
    
    # Azure OpenAI Configuration
    st.subheader("🔧 Azure OpenAI Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Azure OpenAI Endpoint", value=settings.azure_openai_endpoint or "", disabled=True, key="settings_endpoint")
        st.text_input("API Version", value=settings.azure_openai_api_version, disabled=True, key="settings_api_version")
    
    with col2:
        st.text_input("Deployment Name", value=settings.azure_openai_deployment_name, disabled=True, key="settings_deployment")
        st.slider("Max Tokens", value=settings.max_tokens, min_value=1000, max_value=8000, disabled=True, key="settings_max_tokens")
    
    # File Upload Settings
    st.subheader("📁 File Upload Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Max File Size (Standard)", f"{settings.max_file_size_standard / (1024*1024):.0f} MB")
        st.metric("Max File Size (Large)", f"{settings.max_file_size_large / (1024*1024):.0f} MB")
        st.metric("Max Document Length", f"{settings.max_document_length:,} characters")
    
    with col2:
        st.write("**Allowed File Types:**")
        for file_type in settings.allowed_file_types:
            st.write(f"• {file_type}")
    
    # System Information
    st.subheader("ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("App Version", settings.app_version)
        st.metric("Environment", settings.environment)
    
    with col2:
        st.metric("Debug Mode", "Enabled" if settings.debug else "Disabled")
        st.metric("API Port", settings.api_port)
    
    # Clear data button
    st.subheader("🗑️ Data Management")
    
    if st.button("Clear All Analysis Data", type="secondary", key="clear_data_button"):
        st.session_state.documents = []
        st.session_state.current_analysis = None
        st.success("All analysis data has been cleared!")

if __name__ == "__main__":
    main()
