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
from utils.file_utils import validate_file_type, get_file_extension
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
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .analysis-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .entity-tag {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
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
        st.header("🔧 Configuration")
        
        # Document type selection
        document_type = st.selectbox(
            "Document Type",
            ["general", "legal", "financial", "technical", "medical", "business"],
            help="Select the type of document for context-specific analysis",
            key="main_document_type"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        include_entities = st.checkbox("Extract Entities", value=True, key="main_entities")
        include_sentiment = st.checkbox("Sentiment Analysis", value=True, key="main_sentiment")
        include_summary = st.checkbox("Generate Summary", value=True, key="main_summary")
        include_recommendations = st.checkbox("Generate Recommendations", value=True, key="main_recommendations")
        
        # Advanced options
        with st.expander("Advanced Options"):
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
    
    st.header("📤 Upload Document")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a document file",
        type=['pdf', 'docx', 'xlsx', 'txt'],
        help="Supported formats: PDF, DOCX, XLSX, TXT"
    )
    
    if uploaded_file is not None:
        # Display file information
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
        
        with col2:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        
        with col3:
            file_extension = get_file_extension(uploaded_file.name)
            st.metric("File Type", file_extension.upper() if file_extension else "Unknown")
        
        with col4:
            if validate_file_type(uploaded_file.name, uploaded_file.type):
                st.metric("Status", "✅ Valid")
            else:
                st.metric("Status", "❌ Invalid")
        
        # Analyze button
        if st.button("🔍 Analyze Document", type="primary", disabled=not validate_file_type(uploaded_file.name, uploaded_file.type), key="analyze_button"):
            analyze_document(uploaded_file, document_type, include_entities, include_sentiment, include_summary, include_recommendations)

def analyze_document(uploaded_file, document_type, include_entities, include_sentiment, include_summary, include_recommendations):
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
            document_type
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
            'processing_result': processing_result,
            'analysis_result': analysis_result,
            'text_length': len(processing_result['text']),
            'page_count': processing_result.get('page_count', 1)
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
    
    # Summary section
    if analysis.get('summary'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("📝 Summary")
        st.write(analysis['summary'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Key phrases
    if analysis.get('key_phrases'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("🔑 Key Phrases")
        phrases = analysis['key_phrases'][:10]  # Show top 10
        for phrase in phrases:
            st.markdown(f'<span class="entity-tag">{phrase}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Entities
    if analysis.get('entities'):
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
    if analysis.get('sentiment'):
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
    if analysis.get('topics'):
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.subheader("📚 Topics")
        topics = analysis['topics']
        for topic in topics:
            st.write(f"• {topic}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    if analysis.get('recommendations'):
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
    
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.documents:
        st.info("No documents have been analyzed yet. Please upload and analyze some documents first.")
        return
    
    # Overall statistics
    total_docs = len(st.session_state.documents)
    total_pages = sum(doc['page_count'] for doc in st.session_state.documents)
    total_text_length = sum(doc['text_length'] for doc in st.session_state.documents)
    avg_confidence = sum(doc['analysis_result'].get('confidence_score', 0) for doc in st.session_state.documents) / total_docs
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", total_docs)
    
    with col2:
        st.metric("Total Pages", total_pages)
    
    with col3:
        st.metric("Total Text Length", f"{total_text_length:,} chars")
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Document types distribution
    st.subheader("📊 Document Types Distribution")
    doc_types = [doc['document_type'] for doc in st.session_state.documents]
    type_counts = pd.Series(doc_types).value_counts()
    
    fig = px.pie(values=type_counts.values, names=type_counts.index, title="Document Types")
    st.plotly_chart(fig, use_container_width=True)
    
    # File sizes distribution
    st.subheader("📏 File Sizes Distribution")
    file_sizes = [doc['file_size_mb'] for doc in st.session_state.documents]
    
    fig = go.Figure(data=[go.Histogram(x=file_sizes, nbinsx=10)])
    fig.update_layout(title="File Sizes (MB)")
    fig.update_xaxes(title="File Size (MB)")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment analysis over time
    st.subheader("😊 Sentiment Analysis Over Time")
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
        
        fig = px.line(df_sentiment, x='Date', y='Sentiment Score', title="Sentiment Trends")
        st.plotly_chart(fig, use_container_width=True)
    
    # Entity types distribution
    st.subheader("🏷️ Entity Types Distribution")
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
        st.metric("Max File Size", f"{settings.max_file_size / (1024*1024):.0f} MB")
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
