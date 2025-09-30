"""
Document Processing Module
Handles document ingestion, preprocessing, and chunking
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Document processing libraries
import pypdf2
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract

from ..utils.text_utils import TextPreprocessor, TextSplitter

class DocumentChunk:
    """Represents a document chunk with metadata"""
    
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.chunk_id = self._generate_chunk_id()
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID based on content and metadata"""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        doc_id = self.metadata.get('document_id', 'unknown')
        chunk_num = self.metadata.get('chunk_number', 0)
        return f"{doc_id}_{chunk_num}_{content_hash[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            'content': self.content,
            'metadata': self.metadata,
            'chunk_id': self.chunk_id
        }

class DocumentProcessor:
    """Main document processing class"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_preprocessor = TextPreprocessor()
        self.text_splitter = TextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Supported file types
        self.supported_extensions = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_docx,
            '.txt': self._process_txt,
            '.md': self._process_txt,
            '.png': self._process_image,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.tiff': self._process_image
        }
    
    async def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document and return chunks"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check if file type is supported
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {extension}")
            
            # Generate document ID
            document_id = self._generate_document_id(file_path)
            
            # Extract text based on file type
            extractor = self.supported_extensions[extension]
            text = await self._extract_text_async(extractor, file_path)
            
            if not text or not text.strip():
                raise ValueError(f"No text content extracted from {file_path}")
            
            # Preprocess text
            processed_text = await self._preprocess_text_async(text)
            
            # Split into chunks
            chunks = await self._create_chunks_async(processed_text, document_id, file_path.name)
            
            print(f"Processed {file_path.name}: {len(chunks)} chunks created")
            return chunks
            
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            raise
    
    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID"""
        file_content = file_path.read_bytes()
        content_hash = hashlib.md5(file_content).hexdigest()
        return f"doc_{content_hash[:12]}"
    
    async def _extract_text_async(self, extractor, file_path: Path) -> str:
        """Extract text asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, extractor, file_path)
    
    async def _preprocess_text_async(self, text: str) -> str:
        """Preprocess text asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.text_preprocessor.process, text)
    
    async def _create_chunks_async(self, text: str, document_id: str, filename: str) -> List[DocumentChunk]:
        """Create chunks asynchronously"""
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            self.executor,
            self._create_chunks_sync,
            text,
            document_id,
            filename
        )
        return chunks
    
    def _create_chunks_sync(self, text: str, document_id: str, filename: str) -> List[DocumentChunk]:
        """Create chunks synchronously"""
        chunks = []
        text_chunks = self.text_splitter.split_text(text)
        
        for i, chunk_text in enumerate(text_chunks):
            metadata = {
                'document_id': document_id,
                'filename': filename,
                'chunk_number': i,
                'total_chunks': len(text_chunks),
                'chunk_size': len(chunk_text)
            }
            
            chunk = DocumentChunk(chunk_text, metadata)
            chunks.append(chunk)
        
        return chunks
    
    def _process_pdf(self, file_path: Path) -> str:
        """Process PDF file"""
        text = ""
        
        try:
            # Try with pdfplumber first (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"pdfplumber failed, trying PyPDF2: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = pypdf2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"PyPDF2 also failed: {e2}")
                raise ValueError(f"Could not extract text from PDF: {e2}")
        
        return text
    
    def _process_docx(self, file_path: Path) -> str:
        """Process DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text
        except Exception as e:
            raise ValueError(f"Could not extract text from DOCX: {e}")
    
    def _process_txt(self, file_path: Path) -> str:
        """Process text file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Could not decode text file with any supported encoding")
            
        except Exception as e:
            raise ValueError(f"Could not read text file: {e}")
    
    def _process_image(self, file_path: Path) -> str:
        """Process image file using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            raise ValueError(f"Could not extract text from image: {e}")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_extensions.keys())
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False
            
            extension = file_path.suffix.lower()
            return extension in self.supported_extensions
            
        except Exception:
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
