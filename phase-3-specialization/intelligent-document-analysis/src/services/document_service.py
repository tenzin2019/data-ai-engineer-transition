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
    # Check if database is disabled
    if os.getenv("DB_DISABLED", "false").lower() == "true":
        print("✅ Database disabled, skipping initialization")
        return True
    
    # Check if using SQLite and create directory if needed
    database_url = os.getenv("DATABASE_URL", "")
    if "sqlite" in database_url:
        # Extract directory from SQLite path
        import os
        db_path = database_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"✅ Created database directory: {db_dir}")
    
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
    # Check if database is disabled
    if os.getenv("DB_DISABLED", "false").lower() == "true":
        print("✅ Database disabled, skipping save operation")
        return 1  # Return dummy ID
    
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
    # Check if database is disabled
    if os.getenv("DB_DISABLED", "false").lower() == "true":
        print("✅ Database disabled, returning empty list")
        return []
    
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
    # Check if database is disabled
    if os.getenv("DB_DISABLED", "false").lower() == "true":
        print("✅ Database disabled, returning None")
        return None
    
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
    # Check if database is disabled
    if os.getenv("DB_DISABLED", "false").lower() == "true":
        print("✅ Database disabled, skipping delete operation")
        return True
    
    db = next(get_db())
    try:
        from models.document import DocumentAnalysis, DocumentEntity
        
        # Check if document exists first
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            print(f"⚠️ Document {document_id} not found")
            return False
        
        # Delete related records first (due to foreign key constraints)
        analyses_deleted = db.query(DocumentAnalysis).filter(DocumentAnalysis.document_id == document_id).delete()
        entities_deleted = db.query(DocumentEntity).filter(DocumentEntity.document_id == document_id).delete()
        
        # Delete the document
        db.delete(document)
        db.commit()
        
        print(f"✅ Document {document_id} ({document.filename}) and related data deleted successfully!")
        print(f"   - Deleted {analyses_deleted} analyses")
        print(f"   - Deleted {entities_deleted} entities")
        return True
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting document {document_id}: {e}")
        return False
    finally:
        db.close()

def debug_database_state() -> None:
    """Debug function to check database state."""
    db = next(get_db())
    try:
        from sqlalchemy import text
        
        print("🔍 Database State Debug Information:")
        print("=" * 50)
        
        # Check documents
        result = db.execute(text("SELECT COUNT(*) FROM documents"))
        doc_count = result.scalar()
        print(f"Documents: {doc_count}")
        
        if doc_count > 0:
            # Show document details
            result = db.execute(text("SELECT id, filename, status FROM documents LIMIT 5"))
            docs = result.fetchall()
            for doc in docs:
                print(f"  - ID: {doc[0]}, Filename: {doc[1]}, Status: {doc[2]}")
        
        # Check document analyses
        result = db.execute(text("SELECT COUNT(*) FROM document_analyses"))
        analyses_count = result.scalar()
        print(f"Document Analyses: {analyses_count}")
        
        if analyses_count > 0:
            # Show analysis details
            result = db.execute(text("SELECT document_id, analysis_type FROM document_analyses LIMIT 5"))
            analyses = result.fetchall()
            for analysis in analyses:
                print(f"  - Document ID: {analysis[0]}, Type: {analysis[1]}")
        
        # Check document entities
        result = db.execute(text("SELECT COUNT(*) FROM document_entities"))
        entities_count = result.scalar()
        print(f"Document Entities: {entities_count}")
        
        if entities_count > 0:
            # Show entity details
            result = db.execute(text("SELECT document_id, entity_type FROM document_entities LIMIT 5"))
            entities = result.fetchall()
            for entity in entities:
                print(f"  - Document ID: {entity[0]}, Type: {entity[1]}")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error checking database state: {e}")
    finally:
        db.close()

def clear_all_documents_from_db() -> bool:
    """Clear all documents from database using raw SQL for maximum reliability."""
    # Check if database is disabled
    if os.getenv("DB_DISABLED", "false").lower() == "true":
        print("✅ Database disabled, skipping clear operation")
        return True
    
    db = next(get_db())
    try:
        from sqlalchemy import text
        
        # First check if there are any documents to delete
        result = db.execute(text("SELECT COUNT(*) FROM documents"))
        doc_count = result.scalar()
        
        if doc_count == 0:
            print("ℹ️ No documents found in database to clear")
            return True
        
        print(f"🗑️ Clearing {doc_count} documents and related data...")
        
        # Debug database state before clearing
        debug_database_state()
        
        # Method 1: Try raw SQL with explicit deletion order
        try:
            print("   Using raw SQL deletion method...")
            
            # Step 1: Delete all document analyses first
            result1 = db.execute(text("DELETE FROM document_analyses"))
            analyses_deleted = result1.rowcount
            print(f"   ✅ Deleted {analyses_deleted} document analyses")
            
            # Step 2: Delete all document entities second
            result2 = db.execute(text("DELETE FROM document_entities"))
            entities_deleted = result2.rowcount
            print(f"   ✅ Deleted {entities_deleted} document entities")
            
            # Step 3: Delete all documents last
            result3 = db.execute(text("DELETE FROM documents"))
            docs_deleted = result3.rowcount
            print(f"   ✅ Deleted {docs_deleted} documents")
            
            # Commit all changes
            db.commit()
            print(f"✅ Successfully cleared {docs_deleted} documents, {analyses_deleted} analyses, and {entities_deleted} entities!")
            return True
            
        except Exception as sql_error:
            print(f"   ⚠️ Raw SQL method failed: {sql_error}")
            db.rollback()
            
            # Method 2: Try with foreign key constraint handling
            try:
                print("   Trying with foreign key constraint handling...")
                
                # For PostgreSQL, temporarily disable foreign key checks
                db.execute(text("SET session_replication_role = replica;"))
                
                # Delete all records
                result1 = db.execute(text("DELETE FROM document_analyses"))
                analyses_deleted = result1.rowcount
                print(f"   ✅ Deleted {analyses_deleted} document analyses")
                
                result2 = db.execute(text("DELETE FROM document_entities"))
                entities_deleted = result2.rowcount
                print(f"   ✅ Deleted {entities_deleted} document entities")
                
                result3 = db.execute(text("DELETE FROM documents"))
                docs_deleted = result3.rowcount
                print(f"   ✅ Deleted {docs_deleted} documents")
                
                # Re-enable foreign key checks
                db.execute(text("SET session_replication_role = DEFAULT;"))
                
                db.commit()
                print(f"✅ Successfully cleared {docs_deleted} documents, {analyses_deleted} analyses, and {entities_deleted} entities! (FK disabled)")
                return True
                
            except Exception as fk_error:
                print(f"   ⚠️ Foreign key method also failed: {fk_error}")
                db.rollback()
                
                # Method 3: Try individual document deletion with ORM
                try:
                    print("   Trying individual document deletion with ORM...")
                    
                    from models.document import DocumentAnalysis, DocumentEntity
                    
                    documents = db.query(Document).all()
                    docs_deleted = 0
                    analyses_deleted = 0
                    entities_deleted = 0
                    
                    for document in documents:
                        try:
                            # Delete related records for this specific document FIRST
                            analyses_deleted += db.query(DocumentAnalysis).filter(DocumentAnalysis.document_id == document.id).delete()
                            entities_deleted += db.query(DocumentEntity).filter(DocumentEntity.document_id == document.id).delete()
                            
                            # Then delete the document
                            db.delete(document)
                            docs_deleted += 1
                            
                            print(f"   ✅ Deleted document {document.id}: {document.filename}")
                            
                        except Exception as doc_error:
                            print(f"   ⚠️ Error deleting document {document.id}: {doc_error}")
                            continue
                    
                    db.commit()
                    print(f"✅ Successfully cleared {docs_deleted} documents, {analyses_deleted} analyses, and {entities_deleted} entities! (Individual ORM)")
                    return True
                    
                except Exception as orm_error:
                    print(f"   ⚠️ Individual ORM method also failed: {orm_error}")
                    db.rollback()
                    
                    # Method 4: Try TRUNCATE CASCADE (PostgreSQL specific)
                    try:
                        print("   Trying TRUNCATE CASCADE method...")
                        
                        # Use TRUNCATE CASCADE to delete all records at once
                        db.execute(text("TRUNCATE TABLE document_analyses, document_entities, documents CASCADE"))
                        
                        db.commit()
                        print(f"✅ Successfully cleared all data using TRUNCATE CASCADE!")
                        return True
                        
                    except Exception as truncate_error:
                        print(f"   ⚠️ TRUNCATE CASCADE method also failed: {truncate_error}")
                        db.rollback()
                        
                        # Method 5: Try individual document deletion with raw SQL
                        try:
                            print("   Trying individual document deletion with raw SQL...")
                            
                            # Get all document IDs
                            result = db.execute(text("SELECT id FROM documents"))
                            doc_ids = [row[0] for row in result.fetchall()]
                            
                            docs_deleted = 0
                            analyses_deleted = 0
                            entities_deleted = 0
                            
                            for doc_id in doc_ids:
                                try:
                                    # Delete related records for this specific document
                                    result1 = db.execute(text("DELETE FROM document_analyses WHERE document_id = :doc_id"), {"doc_id": doc_id})
                                    analyses_deleted += result1.rowcount
                                    
                                    result2 = db.execute(text("DELETE FROM document_entities WHERE document_id = :doc_id"), {"doc_id": doc_id})
                                    entities_deleted += result2.rowcount
                                    
                                    # Delete the document
                                    result3 = db.execute(text("DELETE FROM documents WHERE id = :doc_id"), {"doc_id": doc_id})
                                    docs_deleted += result3.rowcount
                                    
                                    print(f"   ✅ Deleted document {doc_id}")
                                    
                                except Exception as doc_error:
                                    print(f"   ⚠️ Error deleting document {doc_id}: {doc_error}")
                                    continue
                            
                            db.commit()
                            print(f"✅ Successfully cleared {docs_deleted} documents, {analyses_deleted} analyses, and {entities_deleted} entities! (Individual SQL)")
                            return True
                            
                        except Exception as individual_sql_error:
                            print(f"   ⚠️ Individual SQL method also failed: {individual_sql_error}")
                            db.rollback()
                            return False
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error clearing documents: {e}")
        print(f"   Error details: {str(e)}")
        return False
    finally:
        db.close()
