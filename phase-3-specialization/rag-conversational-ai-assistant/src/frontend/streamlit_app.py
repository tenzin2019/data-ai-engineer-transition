"""
RAG Conversational AI Assistant - Streamlit Frontend
"""

import streamlit as st
import requests
import json
import uuid
from typing import Dict, List, Any, Optional
import time
import os
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Conversational AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

class RAGChatInterface:
    """Main chat interface for the RAG assistant"""
    
    def __init__(self):
        self.api_url = API_BASE_URL
        
        # Initialize session state
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "documents_uploaded" not in st.session_state:
            st.session_state.documents_uploaded = []
    
    def render_header(self):
        """Render the application header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🤖 RAG Conversational AI Assistant")
            st.markdown("**Enterprise-grade RAG system with LLM orchestration**")
    
    def render_sidebar(self):
        """Render the sidebar with controls and information"""
        with st.sidebar:
            st.header("Controls")
            
            # Document upload section
            st.subheader("Document Upload")
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=['pdf', 'docx', 'txt', 'md'],
                accept_multiple_files=True,
                help="Upload documents to add to the knowledge base"
            )
            
            if uploaded_files:
                if st.button("Process Documents", type="primary"):
                    self.upload_documents(uploaded_files)
            
            # Conversation controls
            st.subheader("Conversation")
            if st.button("New Conversation"):
                self.start_new_conversation()
            
            if st.button("Clear Current Chat"):
                self.clear_current_chat()
            
            # Advanced settings
            with st.expander("Advanced Settings"):
                max_tokens = st.slider("Max Tokens", 100, 2000, 500)
                temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
                
                st.session_state.max_tokens = max_tokens
                st.session_state.temperature = temperature
            
            # System information
            with st.expander("System Info"):
                self.show_system_info()
            
            # Documents list
            with st.expander("Knowledge Base"):
                self.show_documents_list()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show sources if available
                    if message["role"] == "assistant" and "sources" in message:
                        self.show_sources(message["sources"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            self.handle_user_input(prompt)
    
    def handle_user_input(self, prompt: str):
        """Handle user input and get assistant response"""
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = self.get_assistant_response(prompt)
                
                if response:
                    st.markdown(response["answer"])
                    
                    # Store assistant message with sources
                    assistant_message = {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Show sources
                    if response.get("sources"):
                        self.show_sources(response["sources"])
                else:
                    error_msg = "Sorry, I encountered an error processing your request."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    def get_assistant_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get response from the RAG assistant API"""
        try:
            # Prepare request
            request_data = {
                "query": query,
                "conversation_id": st.session_state.conversation_id,
                "max_tokens": getattr(st.session_state, 'max_tokens', 500),
                "temperature": getattr(st.session_state, 'temperature', 0.7)
            }
            
            # Make API request
            response = requests.post(
                f"{self.api_url}/query",
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None
    
    def upload_documents(self, uploaded_files):
        """Upload and process documents"""
        try:
            with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                files_data = []
                
                for uploaded_file in uploaded_files:
                    files_data.append(
                        ("files", (uploaded_file.name, uploaded_file.read(), uploaded_file.type))
                    )
                
                # Upload to API
                response = requests.post(
                    f"{self.api_url}/documents/batch",
                    files=files_data,
                    timeout=300  # 5 minutes timeout for large files
                )
                
                if response.status_code == 200:
                    results = response.json()["results"]
                    
                    # Process results
                    success_count = 0
                    for result in results:
                        if result["status"] == "processed":
                            success_count += 1
                            st.session_state.documents_uploaded.append({
                                "filename": result["filename"],
                                "document_id": result["document_id"],
                                "chunks": result["chunks_created"],
                                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                    
                    st.success(f"Successfully processed {success_count}/{len(uploaded_files)} documents")
                    
                    # Show any failures
                    failures = [r for r in results if r["status"] == "failed"]
                    if failures:
                        for failure in failures:
                            st.error(f"Failed to process {failure['filename']}: {failure.get('error', 'Unknown error')}")
                else:
                    st.error(f"Upload failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
    
    def show_sources(self, sources: List[Dict[str, Any]]):
        """Display source documents for the response"""
        if not sources:
            return
        
        with st.expander(f"Sources ({len(sources)} documents)", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**Source {i}: {source.get('filename', 'Unknown')}**")
                st.markdown(f"*Relevance: {source.get('relevance_score', 0):.2f}*")
                
                # Show content preview
                preview = source.get('content_preview', '')
                if preview:
                    st.markdown(f"```\n{preview}\n```")
                
                if i < len(sources):
                    st.divider()
    
    def start_new_conversation(self):
        """Start a new conversation"""
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.success("Started new conversation!")
        st.rerun()
    
    def clear_current_chat(self):
        """Clear current chat messages"""
        try:
            response = requests.delete(f"{self.api_url}/conversations/{st.session_state.conversation_id}")
            if response.status_code == 200:
                st.session_state.messages = []
                st.success("Chat cleared!")
                st.rerun()
            else:
                st.error("Failed to clear chat on server")
        except Exception as e:
            st.error(f"Error clearing chat: {str(e)}")
    
    def show_system_info(self):
        """Show system information"""
        try:
            response = requests.get(f"{self.api_url}/metrics")
            if response.status_code == 200:
                metrics = response.json()
                
                # RAG Engine stats
                rag_stats = metrics.get("rag_engine", {})
                st.write("**RAG Engine:**")
                st.write(f"- Status: {rag_stats.get('engine_status', 'Unknown')}")
                st.write(f"- Active Conversations: {rag_stats.get('active_conversations', 0)}")
                st.write(f"- Total Interactions: {rag_stats.get('total_interactions', 0)}")
                
                # Vector Store stats
                vector_stats = rag_stats.get("vector_store", {})
                if vector_stats:
                    st.write("**Vector Store:**")
                    st.write(f"- Documents: {vector_stats.get('total_documents', 0)}")
                    st.write(f"- Chunks: {vector_stats.get('total_chunks', 0)}")
                
                # LLM Providers
                llm_status = metrics.get("llm_providers", {})
                if llm_status:
                    st.write("**LLM Providers:**")
                    st.write(f"- Available: {llm_status.get('available_providers', 0)}/{llm_status.get('total_providers', 0)}")
            else:
                st.write("Unable to fetch system metrics")
                
        except Exception as e:
            st.write(f"Error fetching system info: {str(e)}")
    
    def show_documents_list(self):
        """Show list of uploaded documents"""
        try:
            response = requests.get(f"{self.api_url}/documents")
            if response.status_code == 200:
                documents = response.json()["documents"]
                
                if documents:
                    for doc in documents:
                        st.write(f"**{doc['filename']}**")
                        st.write(f"- Chunks: {doc['chunk_count']}")
                        st.write(f"- Size: {doc['total_size']} chars")
                        st.write(f"- Created: {doc.get('created_at', 'Unknown')}")
                        
                        # Delete button
                        if st.button(f"Delete", key=f"delete_{doc['document_id']}"):
                            self.delete_document(doc['document_id'], doc['filename'])
                        
                        st.divider()
                else:
                    st.write("No documents uploaded yet")
            else:
                st.write("Unable to fetch documents list")
                
        except Exception as e:
            st.write(f"Error fetching documents: {str(e)}")
    
    def delete_document(self, document_id: str, filename: str):
        """Delete a document"""
        try:
            response = requests.delete(f"{self.api_url}/documents/{document_id}")
            if response.status_code == 200:
                st.success(f"Deleted {filename}")
                st.rerun()
            else:
                st.error(f"Failed to delete {filename}")
        except Exception as e:
            st.error(f"Error deleting document: {str(e)}")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Create layout
        self.render_sidebar()
        
        # Main chat interface
        self.render_chat_interface()

def main():
    """Main application entry point"""
    # Check API connection
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            st.error(f"API not available at {API_BASE_URL}")
            st.stop()
    except requests.exceptions.RequestException:
        st.error(f"Cannot connect to API at {API_BASE_URL}")
        st.info("Make sure the FastAPI backend is running on the correct port.")
        st.stop()
    
    # Initialize and run the chat interface
    chat_interface = RAGChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()
