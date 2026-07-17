"""
Celery tasks for ingestion module.
"""

from celery import Task
from app.core.celery_app import celery_app
from app.services.ingestion.loader import DocumentLoader
from app.services.ingestion.cleaner import process_text
from app.core.database import SessionLocal
from app.models.db_models import Document, Chunk
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.services.ingestion.tasks.process_document")
def process_document_task(self: Task, document_id: int, max_depth: int = 1) -> dict:
    """
    Process uploaded document: load, clean, chunk, and store.
    
    Args:
        document_id: Database ID of document
        max_depth: Crawl depth for URL documents (1=single page, 2=follow links)
        
    Returns:
        Dict with processing results
    """
    db = SessionLocal()
    
    try:
        logger.info(f"=" * 80)
        logger.info(f"Starting document processing for ID: {document_id}")
        
        # Get document
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            error_msg = f"Document {document_id} not found"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Processing document: {document.filename} (type: {document.file_type})")
        logger.info(f"File path: {document.file_path}")
        
        # Load document
        logger.info("Step 1: Loading document...")
        if document.url:
            texts = DocumentLoader.load_url(document.url, max_depth=max_depth)
        else:
            texts = DocumentLoader.load_file(document.file_path)
        
        logger.info(f"Loaded {len(texts)} text segments")
        
        # Combine all texts
        logger.info("Step 2: Combining texts...")
        full_text = "\n\n".join(texts)
        logger.info(f"Combined text length: {len(full_text)} characters")
        
        # Process: clean, chunk, deduplicate
        logger.info("Step 3: Processing text (cleaning, chunking)...")
        chunks = process_text(full_text, enable_deduplication=False)
        logger.info(f"Created {len(chunks)} chunks")
        
        if len(chunks) == 0:
            logger.warning(f"No chunks created! Full text preview: {full_text[:500]}")
        
        # Store chunks in database
        logger.info("Step 4: Storing chunks in database...")
        for idx, (chunk_text, token_count, chunk_hash) in enumerate(chunks):
            chunk = Chunk(
                document_id=document_id,
                chunk_index=idx,
                content=chunk_text,
                token_count=token_count,
                embedding_hash=chunk_hash
            )
            db.add(chunk)
        
        db.commit()
        
        logger.info(f"✅ Successfully stored {len(chunks)} chunks for document {document_id}")
        logger.info(f"=" * 80)
        
        return {
            "status": "success",
            "document_id": document_id,
            "chunk_count": len(chunks),
            "total_tokens": sum(tc for _, tc, _ in chunks)
        }
    
    except Exception as e:
        logger.error(f"❌ Document processing failed for ID {document_id}: {e}")
        logger.exception("Full traceback:")
        db.rollback()
        raise
    
    finally:
        db.close()
