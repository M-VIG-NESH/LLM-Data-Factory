"""
Database models for LLM Data Factory.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.core.database import Base


class JobStatus(str, enum.Enum):
    """Status of generation jobs."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, enum.Enum):
    """Type of generation task."""
    QA_GENERATION = "qa_generation"
    SUMMARIZATION = "summarization"
    CUSTOM = "custom"


class Document(Base):
    """Uploaded document metadata."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, txt, url
    file_path = Column(String(500), nullable=True)
    url = Column(String(1000), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    jobs = relationship("GenerationJob", back_populates="document")


class Chunk(Base):
    """Processed text chunks from documents."""
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order in document
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    embedding_hash = Column(String(64), nullable=True)  # For deduplication
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    class Meta:
        unique_together = ("document_id", "chunk_index")


class GenerationJob(Base):
    """Async generation job tracking."""
    __tablename__ = "generation_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(100), unique=True, index=True, nullable=False)  # Celery task ID
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    task_type = Column(Enum(TaskType), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    dataset_name = Column(String(255), nullable=True)
    few_shot_examples = Column(JSON, nullable=True)  # List of example dicts for few-shot prompting
    
    # Progress tracking
    total_chunks = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Results
    result_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="jobs")
    datasets = relationship("Dataset", back_populates="job", cascade="all, delete-orphan")


class Dataset(Base):
    """Generated training data entries."""
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("generation_jobs.id"), nullable=False)
    
    # Content
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    context = Column(Text, nullable=False)  # Source chunk
    
    # Quality metrics
    complexity_score = Column(Float, nullable=True)
    semantic_similarity = Column(Float, nullable=True)
    toxicity_score = Column(Float, nullable=True)
    is_valid = Column(Integer, default=1)  # Boolean: passed quality checks
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("GenerationJob", back_populates="datasets")
