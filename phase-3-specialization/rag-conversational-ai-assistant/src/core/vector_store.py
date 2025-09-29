"""
Vector Store Module
Handles document embedding and vector storage/retrieval
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime

# Vector database
import chromadb
from chromadb.config import Settings

# Embedding models
from sentence_transformers import SentenceTransformer
import numpy as np

from .document_processor import DocumentChunk

class VectorStore:
    """Vector store implementation using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.collection_name = "rag_documents"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        print(f"✅ Vector store initialized with collection: {self.collection_name}")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(self.collection_name)
            print(f"📚 Using existing collection: {self.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document collection"}
            )
            print(f"🆕 Created new collection: {self.collection_name}")
            return collection
    
    async def add_documents(self, chunks: List[DocumentChunk], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add document chunks to vector store"""
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Add chunk metadata
                chunk_metadata = chunk.metadata.copy()
                if metadata:
                    chunk_metadata.update(metadata)
                
                # Add timestamp
                chunk_metadata['created_at'] = datetime.now().isoformat()
                
                documents.append(chunk.content)
                metadatas.append(chunk_metadata)
                ids.append(chunk.chunk_id)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Return document ID (from first chunk)
            document_id = chunks[0].metadata.get('document_id', str(uuid.uuid4()))
            print(f"✅ Added {len(chunks)} chunks to vector store")
            return document_id
            
        except Exception as e:
            print(f"❌ Error adding documents to vector store: {e}")
            raise
    
    async def search_similar(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0,
                        'id': results['ids'][0][i] if results['ids'] and results['ids'][0] else None
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching vector store: {e}")
            raise
    
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['documents']:
                return None
            
            return {
                'documents': results['documents'],
                'metadatas': results['metadatas'],
                'ids': results['ids']
            }
            
        except Exception as e:
            print(f"❌ Error getting document by ID: {e}")
            return None
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and its chunks"""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['ids']:
                return False
            
            # Delete all chunks
            self.collection.delete(ids=results['ids'])
            
            print(f"✅ Deleted document {document_id} with {len(results['ids'])} chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection"""
        try:
            # Get all documents
            results = self.collection.get()
            
            # Group by document_id
            documents = {}
            
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    doc_id = metadata.get('document_id', 'unknown')
                    
                    if doc_id not in documents:
                        documents[doc_id] = {
                            'document_id': doc_id,
                            'filename': metadata.get('filename', 'unknown'),
                            'created_at': metadata.get('created_at', 'unknown'),
                            'chunk_count': 0,
                            'total_size': 0
                        }
                    
                    documents[doc_id]['chunk_count'] += 1
                    documents[doc_id]['total_size'] += len(doc)
            
            return list(documents.values())
            
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            results = self.collection.get()
            
            total_chunks = len(results['documents']) if results['documents'] else 0
            
            # Count unique documents
            document_ids = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if metadata and 'document_id' in metadata:
                        document_ids.add(metadata['document_id'])
            
            return {
                'total_chunks': total_chunks,
                'total_documents': len(document_ids),
                'collection_name': self.collection_name,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error getting collection stats: {e}")
            return {}
    
    async def update_document_metadata(self, document_id: str, new_metadata: Dict[str, Any]) -> bool:
        """Update document metadata"""
        try:
            # Get existing chunks
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results['ids']:
                return False
            
            # Update metadata for each chunk
            updated_metadatas = []
            for i, metadata in enumerate(results['metadatas']):
                updated_metadata = metadata.copy()
                updated_metadata.update(new_metadata)
                updated_metadatas.append(updated_metadata)
            
            # Update in collection
            self.collection.update(
                ids=results['ids'],
                metadatas=updated_metadatas
            )
            
            print(f"✅ Updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating document metadata: {e}")
            return False
    
    async def search_with_filters(self, query: str, filters: Dict[str, Any], n_results: int = 5) -> List[Dict[str, Any]]:
        """Search with specific filters"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
            
            # Format results
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                        'distance': results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0,
                        'id': results['ids'][0][i] if results['ids'] and results['ids'][0] else None
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching with filters: {e}")
            raise
    
    def reset_collection(self):
        """Reset the entire collection (use with caution)"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            print(f"🔄 Collection {self.collection_name} has been reset")
        except Exception as e:
            print(f"❌ Error resetting collection: {e}")
            raise
