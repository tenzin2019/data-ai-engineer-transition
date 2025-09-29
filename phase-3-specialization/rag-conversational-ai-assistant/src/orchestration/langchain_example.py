"""
LangChain Implementation Example for RAG Conversational AI Assistant

This module demonstrates how to use LangChain for LLM orchestration,
chain management, and RAG implementation.
"""

import os
import asyncio
from typing import Dict, List, Any, Optional
from langchain.llms import OpenAI, Anthropic, AzureOpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic, AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS, Pinecone
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool, Tool
from langchain.agents import initialize_agent, AgentType
import yaml

class LangChainRAGExample:
    """Example implementation of RAG using LangChain."""
    
    def __init__(self, config_path: str = "config/langchain.yaml"):
        self.config = self._load_config(config_path)
        self.models = {}
        self.embeddings = {}
        self.vector_stores = {}
        self.chains = {}
        self.memories = {}
        self.tools = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load LangChain configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default configuration.")
            return {}
    
    def initialize_models(self):
        """Initialize LangChain models based on configuration."""
        print("Initializing LangChain models...")
        
        # Initialize OpenAI models
        if os.getenv("OPENAI_API_KEY"):
            openai_config = self.config.get("langchain", {}).get("models", {}).get("openai", {})
            
            for model_name, config in openai_config.items():
                if model_name == "gpt-4":
                    self.models["openai_gpt4"] = ChatOpenAI(
                        model_name=config["model_name"],
                        temperature=config["temperature"],
                        max_tokens=config["max_tokens"],
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    )
                elif model_name == "gpt-3.5-turbo":
                    self.models["openai_gpt35"] = ChatOpenAI(
                        model_name=config["model_name"],
                        temperature=config["temperature"],
                        max_tokens=config["max_tokens"],
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    )
        
        # Initialize Azure OpenAI models
        if os.getenv("AZURE_OPENAI_API_KEY"):
            azure_config = self.config.get("langchain", {}).get("models", {}).get("azure_openai", {})
            
            for model_name, config in azure_config.items():
                if model_name == "gpt-4":
                    self.models["azure_gpt4"] = AzureChatOpenAI(
                        deployment_name=config["deployment_name"],
                        temperature=config["temperature"],
                        max_tokens=config["max_tokens"],
                        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                    )
        
        # Initialize Anthropic models
        if os.getenv("ANTHROPIC_API_KEY"):
            anthropic_config = self.config.get("langchain", {}).get("models", {}).get("anthropic", {})
            
            for model_name, config in anthropic_config.items():
                if model_name == "claude-3-sonnet":
                    self.models["anthropic_claude3"] = ChatAnthropic(
                        model=config["model"],
                        temperature=config["temperature"],
                        max_tokens_to_sample=config["max_tokens"],
                        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                    )
        
        print(f"Initialized {len(self.models)} models")
    
    def initialize_embeddings(self):
        """Initialize embedding models."""
        print("Initializing embedding models...")
        
        # OpenAI embeddings
        if os.getenv("OPENAI_API_KEY"):
            self.embeddings["openai_ada002"] = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Azure OpenAI embeddings
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self.embeddings["azure_ada002"] = AzureOpenAIEmbeddings(
                deployment="text-embedding-ada-002",
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
        
        # HuggingFace embeddings
        self.embeddings["hf_minilm"] = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        print(f"Initialized {len(self.embeddings)} embedding models")
    
    def create_vector_store(self, documents: List[Document], 
                          vector_store_type: str = "chroma",
                          embedding_model_id: str = "openai_ada002") -> Any:
        """Create vector store from documents."""
        print(f"Creating {vector_store_type} vector store...")
        
        embedding_model = self.embeddings.get(embedding_model_id)
        if not embedding_model:
            raise ValueError(f"Embedding model {embedding_model_id} not found")
        
        if vector_store_type == "chroma":
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory="./chroma_db"
            )
        elif vector_store_type == "faiss":
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embedding_model
            )
        elif vector_store_type == "pinecone":
            import pinecone
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENVIRONMENT")
            )
            
            index_name = os.getenv("PINECONE_INDEX_NAME", "rag-assistant")
            vector_store = Pinecone.from_documents(
                documents=documents,
                embedding=embedding_model,
                index_name=index_name
            )
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
        
        self.vector_stores[vector_store_type] = vector_store
        print(f"Created {vector_store_type} vector store with {len(documents)} documents")
        return vector_store
    
    def create_qa_chain(self, vector_store: Any, 
                       model_id: str = "openai_gpt4",
                       chain_type: str = "stuff") -> RetrievalQA:
        """Create a QA chain for question answering."""
        print(f"Creating QA chain with {model_id}...")
        
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        chain_id = f"qa_{model_id}_{chain_type}"
        self.chains[chain_id] = chain
        print(f"Created QA chain: {chain_id}")
        return chain
    
    def create_conversational_chain(self, vector_store: Any,
                                  model_id: str = "openai_gpt4",
                                  memory_type: str = "buffer") -> ConversationalRetrievalChain:
        """Create a conversational retrieval chain with memory."""
        print(f"Creating conversational chain with {model_id}...")
        
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Create memory
        if memory_type == "buffer":
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        elif memory_type == "summary":
            memory = ConversationSummaryMemory(
                llm=model,
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        else:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        chain_id = f"conversational_{model_id}_{memory_type}"
        self.chains[chain_id] = chain
        self.memories[chain_id] = memory
        print(f"Created conversational chain: {chain_id}")
        return chain
    
    def create_tools(self):
        """Create custom tools for the agent."""
        print("Creating custom tools...")
        
        # Document search tool
        def document_search(query: str) -> str:
            """Search for documents in the knowledge base."""
            # This would integrate with your document search system
            return f"Searching for: {query}"
        
        # Calculator tool
        def calculator(expression: str) -> str:
            """Perform mathematical calculations."""
            try:
                result = eval(expression)
                return str(result)
            except Exception as e:
                return f"Error calculating: {str(e)}"
        
        # Web search tool
        def web_search(query: str) -> str:
            """Search the web for current information."""
            # This would integrate with a web search API
            return f"Web search results for: {query}"
        
        self.tools["document_search"] = Tool(
            name="document_search",
            description="Search for documents in the knowledge base",
            func=document_search
        )
        
        self.tools["calculator"] = Tool(
            name="calculator",
            description="Perform mathematical calculations",
            func=calculator
        )
        
        self.tools["web_search"] = Tool(
            name="web_search",
            description="Search the web for current information",
            func=web_search
        )
        
        print(f"Created {len(self.tools)} tools")
    
    def create_agent(self, model_id: str = "openai_gpt4") -> Any:
        """Create a LangChain agent with tools."""
        print(f"Creating agent with {model_id}...")
        
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        tools = list(self.tools.values())
        
        agent = initialize_agent(
            tools=tools,
            llm=model,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        agent_id = f"agent_{model_id}"
        self.agents = {agent_id: agent}
        print(f"Created agent: {agent_id}")
        return agent
    
    async def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process documents and create embeddings."""
        print(f"Processing {len(file_paths)} documents...")
        
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                print(f"Skipping unsupported file type: {file_path}")
                continue
            
            docs = loader.load()
            documents.extend(docs)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        
        return split_docs
    
    async def query_rag(self, question: str, 
                       chain_id: str = None,
                       model_id: str = "openai_gpt4") -> Dict[str, Any]:
        """Query the RAG system."""
        if not chain_id:
            chain_id = f"qa_{model_id}_stuff"
        
        chain = self.chains.get(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        
        try:
            result = chain({"query": question})
            return {
                "success": True,
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def conversational_query(self, question: str,
                                 chain_id: str = None,
                                 model_id: str = "openai_gpt4") -> Dict[str, Any]:
        """Query the conversational RAG system."""
        if not chain_id:
            chain_id = f"conversational_{model_id}_buffer"
        
        chain = self.chains.get(chain_id)
        if not chain:
            raise ValueError(f"Chain {chain_id} not found")
        
        try:
            result = chain({"question": question})
            return {
                "success": True,
                "answer": result["answer"],
                "chat_history": result.get("chat_history", []),
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        return list(self.models.keys())
    
    def list_available_chains(self) -> List[str]:
        """List all available chains."""
        return list(self.chains.keys())
    
    def list_available_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())

# Example usage
async def main():
    """Example usage of the LangChain RAG implementation."""
    
    # Initialize the RAG system
    rag = LangChainRAGExample()
    
    # Initialize models and embeddings
    rag.initialize_models()
    rag.initialize_embeddings()
    
    # Create tools
    rag.create_tools()
    
    # Process documents (example)
    documents = await rag.process_documents([
        "sample_document.pdf",
        "sample_text.txt"
    ])
    
    # Create vector store
    vector_store = rag.create_vector_store(documents, "chroma")
    
    # Create QA chain
    qa_chain = rag.create_qa_chain(vector_store, "openai_gpt4")
    
    # Create conversational chain
    conv_chain = rag.create_conversational_chain(vector_store, "openai_gpt4")
    
    # Create agent
    agent = rag.create_agent("openai_gpt4")
    
    # Example queries
    questions = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
        "What are the important details mentioned?"
    ]
    
    print("\n=== QA Chain Results ===")
    for question in questions:
        result = await rag.query_rag(question)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print("-" * 50)
    
    print("\n=== Conversational Chain Results ===")
    for question in questions:
        result = await rag.conversational_query(question)
        print(f"Q: {question}")
        print(f"A: {result['answer']}")
        print("-" * 50)
    
    print("\n=== Agent Results ===")
    agent_questions = [
        "What is 2 + 2?",
        "Search for information about artificial intelligence",
        "Calculate the square root of 16"
    ]
    
    for question in agent_questions:
        result = agent.run(question)
        print(f"Q: {question}")
        print(f"A: {result}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
