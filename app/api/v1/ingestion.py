"""
API routes for ingestion endpoints.
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from app.core.database import get_db
from app.core.config import settings
from app.models.schemas import (
    DocumentResponse,
    ChunkResponse,
    UploadResponse,
    URLUploadRequest
)
from app.models.db_models import Document, Chunk
from app.services.ingestion.tasks import process_document_task
from datetime import datetime

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload a document file for processing.
    
    Supported formats: PDF, DOCX, TXT
    """
    # Validate file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB"
        )
    
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.docx', '.doc', '.txt', '.md', '.xlsx', '.xls', '.csv']:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported formats: PDF, DOCX, TXT, MD, XLSX, XLS, CSV"
        )
    
    # Save file
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, file.filename)
    
    with open(file_path, 'wb') as f:
        f.write(content)
    
    # Create database record
    document = Document(
        filename=file.filename,
        file_type=ext[1:],  # Remove dot
        file_path=file_path,
        file_size_bytes=file_size
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Start async processing
    process_document_task.delay(document.id)
    
    return UploadResponse(
        document_id=document.id,
        filename=file.filename,
        message="File uploaded successfully. Processing started."
    )


@router.post("/upload-url", response_model=UploadResponse)
async def upload_url(
    request: URLUploadRequest,
    db: Session = Depends(get_db)
):
    """
    Upload content from a URL for processing.
    """
    filename = request.filename or f"web_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Create database record
    document = Document(
        filename=filename,
        file_type="url",
        url=str(request.url)
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    
    # Start async processing with crawl depth
    process_document_task.delay(document.id, max_depth=request.max_depth)
    
    return UploadResponse(
        document_id=document.id,
        filename=filename,
        message=f"URL submitted successfully. Processing started (depth={request.max_depth})."
    )


@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get list of uploaded documents."""
    documents = db.query(Document).offset(skip).limit(limit).all()
    
    # Add chunk count
    result = []
    for doc in documents:
        chunk_count = db.query(Chunk).filter(Chunk.document_id == doc.id).count()
        doc_dict = {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size_bytes": doc.file_size_bytes,
            "upload_timestamp": doc.upload_timestamp,
            "chunk_count": chunk_count
        }
        result.append(DocumentResponse(**doc_dict))
    
    return result


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get document details."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunk_count = db.query(Chunk).filter(Chunk.document_id == document_id).count()
    
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        file_size_bytes=document.file_size_bytes,
        upload_timestamp=document.upload_timestamp,
        chunk_count=chunk_count
    )


@router.get("/documents/{document_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    document_id: int,
    skip: int = 0,
    limit: int = 2000,
    db: Session = Depends(get_db)
):
    """Get chunks for a document."""
    chunks = db.query(Chunk).filter(
        Chunk.document_id == document_id
    ).order_by(Chunk.chunk_index).offset(skip).limit(limit).all()
    
    return [ChunkResponse.from_orm(chunk) for chunk in chunks]


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document and all its related data."""
    from app.models.db_models import GenerationJob, Dataset
    
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete all generation jobs and their datasets for this document
    jobs = db.query(GenerationJob).filter(GenerationJob.document_id == document_id).all()
    for job in jobs:
        # Delete all datasets for this job
        db.query(Dataset).filter(Dataset.job_id == job.id).delete()
        # Delete the job
        db.delete(job)
    
    # Delete all chunks for this document
    db.query(Chunk).filter(Chunk.document_id == document_id).delete()
    
    # Delete file if exists
    if document.file_path and os.path.exists(document.file_path):
        os.remove(document.file_path)
    
    # Delete the document
    db.delete(document)
    db.commit()
    
    return {"message": "Document and all related data deleted successfully"}
