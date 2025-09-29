"""
RAG Conversational AI Assistant - FastAPI Backend
Main application entry point with API endpoints
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv

from ..core.document_processor import DocumentProcessor
from ..core.vector_store import VectorStore
from ..core.rag_engine import RAGEngine
from ..orchestration.llm_orchestrator import LLMOrchestrator

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Conversational AI Assistant",
    description="Enterprise-grade RAG system with LLM orchestration and monitoring",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (will be initialized properly in production)
document_processor = None
vector_store = None
rag_engine = None
llm_orchestrator = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks_created: int

# Initialize components
async def initialize_components():
    """Initialize all RAG components"""
    global document_processor, vector_store, rag_engine, llm_orchestrator
    
    try:
        # Initialize LLM orchestrator
        llm_orchestrator = LLMOrchestrator()
        
        # Initialize document processor
        document_processor = DocumentProcessor()
        
        # Initialize vector store
        vector_store = VectorStore()
        
        # Initialize RAG engine
        rag_engine = RAGEngine(
            vector_store=vector_store,
            llm_orchestrator=llm_orchestrator
        )
        
        print("✅ All components initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing components: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    await initialize_components()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RAG Conversational AI Assistant API",
        "status": "running",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "document_processor": document_processor is not None,
            "vector_store": vector_store is not None,
            "rag_engine": rag_engine is not None,
            "llm_orchestrator": llm_orchestrator is not None
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Main RAG query endpoint"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Process the query
        result = await rag_engine.process_query(
            query=request.query,
            conversation_id=request.conversation_id,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents"""
    try:
        if not document_processor or not vector_store:
            raise HTTPException(status_code=503, detail="Components not initialized")
        
        # Save uploaded file
        file_path = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document
        chunks = await document_processor.process_document(file_path)
        
        # Store in vector database
        document_id = await vector_store.add_documents(chunks, metadata={"filename": file.filename})
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="processed",
            chunks_created=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        documents = await vector_store.list_documents()
        return {"documents": documents}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        success = await vector_store.delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        history = await rag_engine.get_conversation_history(conversation_id)
        return {"conversation_id": conversation_id, "history": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        # This would be implemented with proper monitoring
        return {
            "total_documents": 0,
            "total_queries": 0,
            "average_response_time": 0.0,
            "system_status": "healthy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
