"""
Document-related database models.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean, DateTime, 
    ForeignKey, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from .base import Base


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Document type enumeration."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    TXT = "txt"
    GENERAL = "general"
    UNKNOWN = "unknown"


class Document(Base):
    """Document model for storing document metadata and content."""
    
    __tablename__ = "documents"
    
    # Basic document information
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    document_type = Column(SQLEnum(DocumentType), nullable=False, default=DocumentType.UNKNOWN)
    
    # Processing status
    status = Column(SQLEnum(DocumentStatus), nullable=False, default=DocumentStatus.UPLOADED)
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Extracted content
    extracted_text = Column(Text, nullable=True)
    text_length = Column(Integer, nullable=True)
    page_count = Column(Integer, nullable=True)
    
    # AI Analysis results
    summary = Column(Text, nullable=True)
    key_phrases = Column(JSON, nullable=True)  # List of key phrases
    sentiment_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Metadata
    document_metadata = Column(JSON, nullable=True)  # Additional document metadata
    tags = Column(JSON, nullable=True)  # User-defined tags
    
    # User relationship
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="documents")
    
    # Analysis relationship
    analyses = relationship("DocumentAnalysis", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("DocumentEntity", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"


class DocumentAnalysis(Base):
    """Document analysis results model."""
    
    __tablename__ = "document_analyses"
    
    # Document relationship
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    document = relationship("Document", back_populates="analyses")
    
    # Analysis type and results
    analysis_type = Column(String(100), nullable=False)  # e.g., "summarization", "classification"
    analysis_data = Column(JSON, nullable=False)  # Analysis results in JSON format
    
    # AI model information
    model_name = Column(String(100), nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Performance metrics
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    token_count = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)  # API cost if applicable
    
    # Quality metrics
    confidence_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    
    def __repr__(self) -> str:
        return f"<DocumentAnalysis(id={self.id}, document_id={self.document_id}, type='{self.analysis_type}')>"


class DocumentEntity(Base):
    """Extracted entities from documents."""
    
    __tablename__ = "document_entities"
    
    # Document relationship
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    document = relationship("Document", back_populates="entities")
    
    # Entity information
    entity_text = Column(String(500), nullable=False)
    entity_type = Column(String(100), nullable=False)  # e.g., "PERSON", "ORGANIZATION", "DATE"
    entity_label = Column(String(100), nullable=True)  # Custom label
    
    # Position information
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    page_number = Column(Integer, nullable=True)
    
    # Confidence and metadata
    confidence_score = Column(Float, nullable=True)
    entity_metadata = Column(JSON, nullable=True)  # Additional entity information
    
    def __repr__(self) -> str:
        return f"<DocumentEntity(id={self.id}, text='{self.entity_text}', type='{self.entity_type}')>"
