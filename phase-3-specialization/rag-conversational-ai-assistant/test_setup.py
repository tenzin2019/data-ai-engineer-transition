#!/usr/bin/env python3
"""
RAG Conversational AI Assistant - Setup Test
Quick test to verify the system is working correctly
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.rag_engine import RAGEngine
from orchestration.llm_orchestrator import LLMOrchestrator
from utils.text_utils import TextPreprocessor, TextSplitter


async def test_setup():
    """Test the RAG system setup"""
    print("🤖 RAG Conversational AI Assistant - Setup Test")
    print("=" * 60)
    
    # Test 1: Text Utilities
    print("\n1. Testing Text Utilities...")
    try:
        preprocessor = TextPreprocessor()
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        
        test_text = "This is a test document. It contains multiple sentences. We will test text processing."
        processed = preprocessor.process(test_text)
        chunks = splitter.split_text(processed)
        
        print(f"   ✅ Processed text: {len(processed)} characters")
        print(f"   ✅ Created chunks: {len(chunks)}")
        
    except Exception as e:
        print(f"   ❌ Text utilities failed: {e}")
        return False
    
    # Test 2: Document Processor
    print("\n2. Testing Document Processor...")
    try:
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        
        # Create a test text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write("This is a test document for the RAG system. " * 10)
            temp_file_path = temp_file.name
        
        # Process the document
        chunks = await processor.process_document(temp_file_path)
        print(f"   ✅ Document processed: {len(chunks)} chunks created")
        
        # Cleanup
        os.unlink(temp_file_path)
        await processor.cleanup()
        
    except Exception as e:
        print(f"   ❌ Document processor failed: {e}")
        return False
    
    # Test 3: Vector Store
    print("\n3. Testing Vector Store...")
    try:
        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(persist_directory=temp_dir)
            
            # Add some test chunks
            if chunks:
                doc_id = await vector_store.add_documents(chunks[:2])  # Add first 2 chunks
                print(f"   ✅ Documents added to vector store: {doc_id}")
                
                # Test search
                results = await vector_store.search_similar("test document", n_results=1)
                print(f"   ✅ Search results: {len(results)} documents found")
                
                # Test stats
                stats = await vector_store.get_collection_stats()
                print(f"   ✅ Collection stats: {stats.get('total_chunks', 0)} chunks")
        
    except Exception as e:
        print(f"   ❌ Vector store failed: {e}")
        return False
    
    # Test 4: LLM Orchestrator (without actual API calls)
    print("\n4. Testing LLM Orchestrator...")
    try:
        orchestrator = LLMOrchestrator()
        status = await orchestrator.get_provider_status()
        print(f"   ✅ LLM orchestrator initialized")
        print(f"   ℹ️ Available providers: {status.get('available_providers', 0)}")
        
        if status.get('available_providers', 0) == 0:
            print("   ⚠️ No LLM providers configured (set API keys to test)")
        
    except Exception as e:
        print(f"   ❌ LLM orchestrator failed: {e}")
        return False
    
    # Test 5: RAG Engine Integration (without LLM calls)
    print("\n5. Testing RAG Engine...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(persist_directory=temp_dir)
            orchestrator = LLMOrchestrator()
            
            rag_engine = RAGEngine(
                vector_store=vector_store,
                llm_orchestrator=orchestrator
            )
            
            # Add test documents
            if chunks:
                await vector_store.add_documents(chunks)
            
            # Test stats
            stats = await rag_engine.get_engine_stats()
            print(f"   ✅ RAG engine initialized")
            print(f"   ✅ Engine status: {stats.get('engine_status', 'unknown')}")
        
    except Exception as e:
        print(f"   ❌ RAG engine failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All core components are working correctly!")
    print("\n🚀 Next steps:")
    print("1. Set up your API keys in the .env file")
    print("2. Run: python src/api/main.py")
    print("3. In another terminal: streamlit run src/frontend/streamlit_app.py")
    print("4. Or use Docker: docker-compose up -d")
    
    return True


def test_api_keys():
    """Test if API keys are configured"""
    print("\n🔑 Checking API Key Configuration:")
    
    keys_to_check = [
        ("OPENAI_API_KEY", "OpenAI"),
        ("ANTHROPIC_API_KEY", "Anthropic"),
        ("AZURE_OPENAI_API_KEY", "Azure OpenAI"),
        ("AZURE_OPENAI_ENDPOINT", "Azure OpenAI Endpoint")
    ]
    
    configured_count = 0
    
    for env_var, service in keys_to_check:
        value = os.getenv(env_var)
        if value and value != "your-api-key-here" and value != "":
            print(f"   ✅ {service}: Configured")
            configured_count += 1
        else:
            print(f"   ⚠️ {service}: Not configured")
    
    if configured_count == 0:
        print("\n   ℹ️ No API keys configured. Set them in .env file to test LLM functionality.")
    else:
        print(f"\n   ✅ {configured_count}/{len(keys_to_check)} services configured")
    
    return configured_count > 0


def test_dependencies():
    """Test if all required dependencies are installed"""
    print("\n📦 Checking Dependencies:")
    
    dependencies = [
        ("fastapi", "FastAPI web framework"),
        ("uvicorn", "ASGI server"),
        ("streamlit", "Frontend framework"),
        ("chromadb", "Vector database"),
        ("sentence_transformers", "Embedding models"),
        ("openai", "OpenAI client"),
        ("anthropic", "Anthropic client"),
        ("langchain", "LLM orchestration"),
        ("pydantic", "Data validation"),
        ("sqlalchemy", "Database ORM")
    ]
    
    missing_deps = []
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"   ✅ {package}: {description}")
        except ImportError:
            print(f"   ❌ {package}: Missing - {description}")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n   ⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n   ✅ All dependencies are installed")
        return True


async def main():
    """Main test function"""
    print("Starting RAG Conversational AI Assistant setup test...\n")
    
    # Test dependencies first
    deps_ok = test_dependencies()
    if not deps_ok:
        print("\n❌ Please install missing dependencies first.")
        return
    
    # Test API keys
    test_api_keys()
    
    # Test system components
    system_ok = await test_setup()
    
    if system_ok:
        print("\n🎉 Setup test completed successfully!")
        print("The RAG Conversational AI Assistant is ready to use.")
    else:
        print("\n❌ Setup test failed. Please check the errors above.")


if __name__ == "__main__":
    asyncio.run(main())