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
import time
import psutil
from datetime import datetime
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
        
        print("All components initialized successfully")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
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
    """Comprehensive health check endpoint"""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "uptime": int(time.time()),  # Will be properly calculated in production
        "components": {
            "document_processor": {
                "status": "healthy" if document_processor is not None else "unhealthy",
                "initialized": document_processor is not None
            },
            "vector_store": {
                "status": "healthy" if vector_store is not None else "unhealthy",
                "initialized": vector_store is not None
            },
            "rag_engine": {
                "status": "healthy" if rag_engine is not None else "unhealthy",
                "initialized": rag_engine is not None
            },
            "llm_orchestrator": {
                "status": "healthy" if llm_orchestrator is not None else "unhealthy",
                "initialized": llm_orchestrator is not None
            }
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            }
        },
        "configuration": {
            "database_url_configured": bool(os.getenv("DATABASE_URL")),
            "redis_url_configured": bool(os.getenv("REDIS_URL")),
            "openai_api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic_api_key_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
            "azure_openai_configured": bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
        }
    }
    
    # Check if any critical components are unhealthy
    unhealthy_components = [
        comp for comp, data in health_status["components"].items() 
        if data["status"] == "unhealthy"
    ]
    
    if unhealthy_components:
        health_status["status"] = "degraded"
        health_status["issues"] = f"Unhealthy components: {', '.join(unhealthy_components)}"
    
    # Return appropriate HTTP status
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes-style readiness probe"""
    global document_processor, vector_store, rag_engine, llm_orchestrator
    
    ready = all([
        document_processor is not None,
        vector_store is not None,
        rag_engine is not None,
        llm_orchestrator is not None
    ])
    
    if ready:
        return {"status": "ready", "message": "All components initialized"}
    else:
        return JSONResponse(
            content={"status": "not_ready", "message": "Components still initializing"},
            status_code=503
        )

@app.get("/health/live")
async def liveness_check():
    """Kubernetes-style liveness probe"""
    return {"status": "alive", "timestamp": int(time.time())}

@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint"""
    global rag_engine, llm_orchestrator
    
    metrics = {
        "rag_queries_total": 0,  # Will be implemented with proper metrics tracking
        "rag_queries_successful": 0,
        "rag_queries_failed": 0,
        "documents_processed_total": 0,
        "vector_store_documents": 0,
        "llm_requests_total": 0,
        "avg_response_time_seconds": 0.0,
        "system_info": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
    
    # Get actual metrics from components if available
    try:
        if rag_engine:
            engine_stats = await rag_engine.get_engine_stats()
            metrics.update(engine_stats)
    except Exception as e:
        print(f"Error getting engine stats: {e}")
    
    try:
        if vector_store:
            collection_stats = await vector_store.get_collection_stats()
            metrics["vector_store_documents"] = collection_stats.get("documents", 0)
    except Exception as e:
        print(f"Error getting collection stats: {e}")
    
    return metrics

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

@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        success = await rag_engine.clear_conversation(conversation_id)
        
        if success:
            return {"message": f"Conversation {conversation_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Get RAG engine stats
        rag_stats = await rag_engine.get_engine_stats()
        
        # Get LLM orchestrator status
        llm_status = await llm_orchestrator.get_provider_status() if llm_orchestrator else {}
        
        return {
            "rag_engine": rag_stats,
            "llm_providers": llm_status,
            "system_status": "healthy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/providers/reset")
async def reset_provider_errors(provider_name: Optional[str] = None):
    """Reset errors for LLM providers"""
    try:
        if not llm_orchestrator:
            raise HTTPException(status_code=503, detail="LLM orchestrator not initialized")
        
        await llm_orchestrator.reset_provider_errors(provider_name)
        return {"message": f"Reset errors for provider: {provider_name or 'all providers'}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/providers/status")
async def get_provider_status():
    """Get status of all LLM providers"""
    try:
        if not llm_orchestrator:
            raise HTTPException(status_code=503, detail="LLM orchestrator not initialized")
        
        status = await llm_orchestrator.get_provider_status()
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/batch")
async def upload_batch_documents(files: List[UploadFile] = File(...)):
    """Upload and process multiple documents"""
    try:
        if not document_processor or not vector_store:
            raise HTTPException(status_code=503, detail="Components not initialized")
        
        results = []
        
        for file in files:
            try:
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
                
                results.append({
                    "document_id": document_id,
                    "filename": file.filename,
                    "status": "processed",
                    "chunks_created": len(chunks)
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(query: str, limit: int = 5):
    """Search documents without generating a response"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        results = await vector_store.search_similar(query, n_results=limit)
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
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
