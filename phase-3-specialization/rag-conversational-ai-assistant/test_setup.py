#!/usr/bin/env python3
"""
Test script to verify RAG system setup and functionality
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

async def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from src.core.document_processor import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
        
        from src.core.vector_store import VectorStore
        print("✅ VectorStore imported successfully")
        
        from src.core.rag_engine import RAGEngine
        print("✅ RAGEngine imported successfully")
        
        from src.orchestration.llm_orchestrator import LLMOrchestrator
        print("✅ LLMOrchestrator imported successfully")
        
        from src.utils.text_utils import TextPreprocessor, TextSplitter
        print("✅ Text utilities imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

async def test_text_processing():
    """Test text processing functionality"""
    print("\n🧪 Testing text processing...")
    
    try:
        from src.utils.text_utils import TextPreprocessor, TextSplitter
        
        # Test text preprocessing
        preprocessor = TextPreprocessor()
        test_text = "  Hello   World!   This   is   a   test.  "
        processed = preprocessor.process(test_text)
        print(f"✅ Text preprocessing: '{test_text}' -> '{processed}'")
        
        # Test text splitting
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        long_text = "This is a long text that should be split into multiple chunks for processing. " * 5
        chunks = splitter.split_text(long_text)
        print(f"✅ Text splitting: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing error: {e}")
        return False

async def test_document_processor():
    """Test document processor initialization"""
    print("\n🧪 Testing document processor...")
    
    try:
        from src.core.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        supported_formats = processor.get_supported_formats()
        print(f"✅ Document processor initialized with formats: {supported_formats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor error: {e}")
        return False

async def test_vector_store():
    """Test vector store initialization"""
    print("\n🧪 Testing vector store...")
    
    try:
        from src.core.vector_store import VectorStore
        
        vector_store = VectorStore(persist_directory="./test_chroma_db")
        stats = await vector_store.get_collection_stats()
        print(f"✅ Vector store initialized: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store error: {e}")
        return False

async def test_llm_orchestrator():
    """Test LLM orchestrator initialization"""
    print("\n🧪 Testing LLM orchestrator...")
    
    try:
        from src.orchestration.llm_orchestrator import LLMOrchestrator
        
        orchestrator = LLMOrchestrator()
        status = await orchestrator.get_provider_status()
        print(f"✅ LLM orchestrator initialized: {status}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM orchestrator error: {e}")
        return False

async def test_rag_engine():
    """Test RAG engine initialization"""
    print("\n🧪 Testing RAG engine...")
    
    try:
        from src.core.vector_store import VectorStore
        from src.orchestration.llm_orchestrator import LLMOrchestrator
        from src.core.rag_engine import RAGEngine
        
        vector_store = VectorStore(persist_directory="./test_chroma_db")
        llm_orchestrator = LLMOrchestrator()
        rag_engine = RAGEngine(vector_store, llm_orchestrator)
        
        stats = await rag_engine.get_engine_stats()
        print(f"✅ RAG engine initialized: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG engine error: {e}")
        return False

async def test_fastapi_app():
    """Test FastAPI app import"""
    print("\n🧪 Testing FastAPI app...")
    
    try:
        from src.api.main import app
        print("✅ FastAPI app imported successfully")
        
        # Test app endpoints
        routes = [route.path for route in app.routes]
        print(f"✅ Available routes: {routes}")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting RAG System Setup Tests\n")
    
    tests = [
        ("Import Test", test_imports),
        ("Text Processing", test_text_processing),
        ("Document Processor", test_document_processor),
        ("Vector Store", test_vector_store),
        ("LLM Orchestrator", test_llm_orchestrator),
        ("RAG Engine", test_rag_engine),
        ("FastAPI App", test_fastapi_app),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Set up your API keys in .env file")
        print("2. Run: python -m uvicorn src.api.main:app --reload")
        print("3. Open http://localhost:8000 to test the API")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    # Cleanup test database
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")
        print("🧹 Cleaned up test database")

if __name__ == "__main__":
    asyncio.run(main())

