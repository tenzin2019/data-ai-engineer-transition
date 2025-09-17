"""
Document service for database operations.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database import get_db, create_tables
from models.document import Document, DocumentAnalysis, DocumentEntity, DocumentStatus, DocumentType
from sqlalchemy.orm import Session

def init_database():
    """Initialize database tables."""
    return create_tables()

def save_document_to_db(
    filename: str,
    original_filename: str,
    file_path: str,
    file_size: int,
    mime_type: str,
    document_type: str,
    extracted_text: str = None,
    text_length: int = None,
    page_count: int = None,
    summary: str = None,
    key_phrases: List[str] = None,
    sentiment_score: float = None,
    confidence_score: float = None,
    analysis_data: Dict = None,
    user_id: int = None
) -> int:
    """Save document to database and return document ID."""
    
    db = next(get_db())
    try:
        # Create document record
        document = Document(
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            mime_type=mime_type,
            document_type=DocumentType(document_type.lower()) if document_type else DocumentType.UNKNOWN,
            status=DocumentStatus.COMPLETED,
            extracted_text=extracted_text,
            text_length=text_length,
            page_count=page_count,
            summary=summary,
            key_phrases=key_phrases,
            sentiment_score=sentiment_score,
            confidence_score=confidence_score,
            user_id=user_id
        )
        
        db.add(document)
        db.commit()
        db.refresh(document)
        
        # Save analysis data if provided
        if analysis_data:
            analysis = DocumentAnalysis(
                document_id=document.id,
                analysis_type="comprehensive_analysis",
                analysis_data=analysis_data,
                model_name="streamlit_app",
                model_version="1.0",
                confidence_score=confidence_score
            )
            db.add(analysis)
            db.commit()
        
        return document.id
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def get_documents_from_db(limit: int = 100) -> List[Dict]:
    """Get documents from database."""
    
    db = next(get_db())
    try:
        # Use raw SQL to avoid enum conversion issues
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT d.id, d.filename, d.original_filename, d.file_path, d.file_size, d.mime_type, 
                   d.document_type, d.status, d.created_at, d.updated_at, d.processing_started_at,
                   d.processing_completed_at, d.error_message, d.extracted_text, d.text_length,
                   d.page_count, d.summary, d.key_phrases, d.sentiment_score, d.confidence_score,
                   d.document_metadata, d.tags, d.user_id,
                   da.analysis_data
            FROM documents d
            LEFT JOIN document_analyses da ON d.id = da.document_id AND da.analysis_type = 'comprehensive_analysis'
            ORDER BY d.created_at DESC 
            LIMIT :limit
        """), {"limit": limit})
        
        documents = result.fetchall()
        
        result_list = []
        for doc in documents:
            # Handle document_type safely
            try:
                if hasattr(doc.document_type, 'value'):
                    doc_type = doc.document_type.value
                else:
                    doc_type = str(doc.document_type) if doc.document_type else 'unknown'
            except:
                doc_type = 'unknown'
            
            # Get analysis data from document_analyses table
            analysis_data = doc.analysis_data if doc.analysis_data else {
                'summary': doc.summary,
                'key_phrases': doc.key_phrases or [],
                'sentiment': {
                    'score': doc.sentiment_score,
                    'label': 'positive' if doc.sentiment_score and doc.sentiment_score > 0.1 else 'negative' if doc.sentiment_score and doc.sentiment_score < -0.1 else 'neutral'
                } if doc.sentiment_score else {},
                'confidence_score': doc.confidence_score,
                'entities': []
            }
            
            result_list.append({
                'id': doc.id,
                'filename': doc.filename,
                'original_filename': doc.original_filename,
                'file_type': doc_type,
                'document_type': doc_type,
                'file_size': doc.file_size,
                'file_size_mb': round(doc.file_size / (1024 * 1024), 2),
                'mime_type': doc.mime_type,
                'status': doc.status if doc.status else 'unknown',
                'upload_time': doc.created_at,
                'page_count': doc.page_count,
                'text_length': doc.text_length,
                'summary': analysis_data.get('summary', doc.summary),
                'key_phrases': analysis_data.get('key_phrases', doc.key_phrases or []),
                'sentiment_score': analysis_data.get('sentiment', {}).get('score', doc.sentiment_score),
                'confidence_score': analysis_data.get('confidence_score', doc.confidence_score),
                'analysis_result': analysis_data
            })
        
        return result_list
        
    except Exception as e:
        print(f"Error getting documents from database: {e}")
        return []
    finally:
        db.close()

def get_document_by_id(document_id: int) -> Optional[Dict]:
    """Get specific document by ID."""
    
    db = next(get_db())
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            return None
            
        # Handle document_type safely
        try:
            if hasattr(document.document_type, 'value'):
                doc_type = document.document_type.value
            else:
                doc_type = str(document.document_type)
        except:
            # Fallback to raw value from database
            doc_type = getattr(document, 'document_type', 'unknown')
            if hasattr(doc_type, 'value'):
                doc_type = doc_type.value
            else:
                doc_type = str(doc_type)
        
        # Get analysis data from document_analyses table
        analysis_data = {}
        analysis_record = db.query(DocumentAnalysis).filter(
            DocumentAnalysis.document_id == document_id,
            DocumentAnalysis.analysis_type == "comprehensive_analysis"
        ).first()
        
        if analysis_record and analysis_record.analysis_data:
            analysis_data = analysis_record.analysis_data
        else:
            # Fallback to document fields
            analysis_data = {
                'summary': document.summary,
                'key_phrases': document.key_phrases or [],
                'sentiment': {
                    'score': document.sentiment_score,
                    'label': 'positive' if document.sentiment_score and document.sentiment_score > 0.1 else 'negative' if document.sentiment_score and document.sentiment_score < -0.1 else 'neutral'
                } if document.sentiment_score else {},
                'confidence_score': document.confidence_score,
                'entities': []
            }
            
        return {
            'id': document.id,
            'filename': document.filename,
            'original_filename': document.original_filename,
            'file_type': doc_type,
            'document_type': doc_type,
            'file_size': document.file_size,
            'file_size_mb': round(document.file_size / (1024 * 1024), 2),
            'mime_type': document.mime_type,
            'status': document.status.value,
            'upload_time': document.created_at,
            'page_count': document.page_count,
            'text_length': document.text_length,
            'summary': analysis_data.get('summary', document.summary),
            'key_phrases': analysis_data.get('key_phrases', document.key_phrases or []),
            'sentiment_score': analysis_data.get('sentiment', {}).get('score', document.sentiment_score),
            'confidence_score': analysis_data.get('confidence_score', document.confidence_score),
            'analysis_result': analysis_data
        }
        
    except Exception as e:
        print(f"Error getting document by ID: {e}")
        return None
    finally:
        db.close()

def delete_document_from_db(document_id: int) -> bool:
    """Delete document from database."""
    
    db = next(get_db())
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if document:
            db.delete(document)
            db.commit()
            return True
        return False
        
    except Exception as e:
        db.rollback()
        print(f"Error deleting document: {e}")
        return False
    finally:
        db.close()

def clear_all_documents_from_db() -> bool:
    """Clear all documents from database."""
    
    db = next(get_db())
    try:
        db.query(Document).delete()
        db.commit()
        return True
        
    except Exception as e:
        db.rollback()
        print(f"Error clearing documents: {e}")
        return False
    finally:
        db.close()
