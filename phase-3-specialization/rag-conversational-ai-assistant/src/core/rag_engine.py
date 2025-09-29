"""
RAG Engine Module
Main orchestration engine for retrieval-augmented generation
"""

import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from .vector_store import VectorStore
from ..orchestration.llm_orchestrator import LLMOrchestrator
from ..utils.text_utils import TextPreprocessor

class ConversationMemory:
    """Manages conversation context and memory"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_interaction(self, conversation_id: str, query: str, response: str, sources: List[Dict[str, Any]]):
        """Add interaction to conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'sources': sources
        }
        
        self.conversations[conversation_id].append(interaction)
        
        # Keep only last 10 interactions to manage memory
        if len(self.conversations[conversation_id]) > 10:
            self.conversations[conversation_id] = self.conversations[conversation_id][-10:]
    
    def get_context(self, conversation_id: str, max_interactions: int = 3) -> str:
        """Get conversation context for the LLM"""
        if conversation_id not in self.conversations:
            return ""
        
        recent_interactions = self.conversations[conversation_id][-max_interactions:]
        context_parts = []
        
        for interaction in recent_interactions:
            context_parts.append(f"User: {interaction['query']}")
            context_parts.append(f"Assistant: {interaction['response']}")
        
        return "\n".join(context_parts)
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get full conversation history"""
        return self.conversations.get(conversation_id, [])

class RAGEngine:
    """Main RAG engine that orchestrates retrieval and generation"""
    
    def __init__(self, vector_store: VectorStore, llm_orchestrator: LLMOrchestrator):
        self.vector_store = vector_store
        self.llm_orchestrator = llm_orchestrator
        self.text_preprocessor = TextPreprocessor()
        self.conversation_memory = ConversationMemory()
        
        # RAG configuration
        self.max_context_length = 4000
        self.default_n_results = 5
        self.context_window = 3
        
        print("✅ RAG Engine initialized")
    
    async def process_query(self, query: str, conversation_id: Optional[str] = None, 
                          max_tokens: int = 500, temperature: float = 0.7) -> Dict[str, Any]:
        """Process a query through the RAG pipeline"""
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Step 1: Preprocess query
            processed_query = await self._preprocess_query(query)
            
            # Step 2: Retrieve relevant documents
            retrieved_docs = await self._retrieve_documents(processed_query)
            
            # Step 3: Build context
            context = await self._build_context(retrieved_docs, conversation_id)
            
            # Step 4: Generate response
            response = await self._generate_response(
                query=processed_query,
                context=context,
                conversation_id=conversation_id,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Step 5: Extract sources
            sources = await self._extract_sources(retrieved_docs)
            
            # Step 6: Store conversation
            self.conversation_memory.add_interaction(
                conversation_id=conversation_id,
                query=query,
                response=response,
                sources=sources
            )
            
            # Step 7: Prepare response
            result = {
                'answer': response,
                'sources': sources,
                'conversation_id': conversation_id,
                'metadata': {
                    'query_processed': processed_query,
                    'documents_retrieved': len(retrieved_docs),
                    'context_length': len(context),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            print(f"✅ Query processed successfully: {conversation_id}")
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            raise
    
    async def _preprocess_query(self, query: str) -> str:
        """Preprocess the user query"""
        try:
            # Clean and normalize the query
            processed = self.text_preprocessor.process(query)
            
            # Could add more sophisticated preprocessing here:
            # - Query expansion
            # - Intent classification
            # - Entity extraction
            
            return processed
            
        except Exception as e:
            print(f"❌ Error preprocessing query: {e}")
            return query  # Return original query as fallback
    
    async def _retrieve_documents(self, query: str, n_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector store"""
        try:
            if n_results is None:
                n_results = self.default_n_results
            
            # Perform similarity search
            results = await self.vector_store.search_similar(query, n_results=n_results)
            
            # Filter and rank results
            filtered_results = []
            for result in results:
                # Filter out very low similarity scores
                if result.get('distance', 1.0) < 0.9:  # ChromaDB uses cosine distance
                    filtered_results.append(result)
            
            print(f"📚 Retrieved {len(filtered_results)} relevant documents")
            return filtered_results
            
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []
    
    async def _build_context(self, retrieved_docs: List[Dict[str, Any]], conversation_id: str) -> str:
        """Build context from retrieved documents and conversation history"""
        try:
            context_parts = []
            
            # Add conversation history
            conversation_context = self.conversation_memory.get_context(
                conversation_id, 
                max_interactions=self.context_window
            )
            
            if conversation_context:
                context_parts.append("Previous conversation:")
                context_parts.append(conversation_context)
                context_parts.append("")  # Empty line for separation
            
            # Add retrieved documents
            if retrieved_docs:
                context_parts.append("Relevant documents:")
                
                for i, doc in enumerate(retrieved_docs, 1):
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    filename = metadata.get('filename', 'Unknown')
                    
                    context_parts.append(f"Document {i} ({filename}):")
                    context_parts.append(content)
                    context_parts.append("")  # Empty line between documents
            
            # Join and truncate if necessary
            context = "\n".join(context_parts)
            
            if len(context) > self.max_context_length:
                # Truncate context while preserving structure
                context = context[:self.max_context_length]
                # Find the last complete document or conversation turn
                last_newline = context.rfind('\n')
                if last_newline > self.max_context_length * 0.8:  # Keep most of the context
                    context = context[:last_newline]
            
            return context
            
        except Exception as e:
            print(f"❌ Error building context: {e}")
            return ""
    
    async def _generate_response(self, query: str, context: str, conversation_id: str,
                               max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using LLM"""
        try:
            # Build the prompt
            prompt = self._build_prompt(query, context)
            
            # Generate response using LLM orchestrator
            response = await self.llm_orchestrator.generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                conversation_id=conversation_id
            )
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build the prompt for the LLM"""
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. Use the information from the context to provide accurate and helpful answers. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    async def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from retrieved documents"""
        try:
            sources = []
            
            for doc in retrieved_docs:
                metadata = doc.get('metadata', {})
                source = {
                    'filename': metadata.get('filename', 'Unknown'),
                    'document_id': metadata.get('document_id', 'Unknown'),
                    'chunk_id': doc.get('id', 'Unknown'),
                    'relevance_score': 1.0 - doc.get('distance', 1.0),  # Convert distance to similarity
                    'content_preview': doc.get('content', '')[:200] + '...' if len(doc.get('content', '')) > 200 else doc.get('content', '')
                }
                sources.append(source)
            
            return sources
            
        except Exception as e:
            print(f"❌ Error extracting sources: {e}")
            return []
    
    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_memory.get_conversation_history(conversation_id)
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history"""
        try:
            if conversation_id in self.conversation_memory.conversations:
                del self.conversation_memory.conversations[conversation_id]
                return True
            return False
        except Exception as e:
            print(f"❌ Error clearing conversation: {e}")
            return False
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        try:
            vector_stats = await self.vector_store.get_collection_stats()
            
            return {
                'vector_store': vector_stats,
                'active_conversations': len(self.conversation_memory.conversations),
                'total_interactions': sum(len(conv) for conv in self.conversation_memory.conversations.values()),
                'engine_status': 'healthy'
            }
            
        except Exception as e:
            print(f"❌ Error getting engine stats: {e}")
            return {'engine_status': 'error'}
