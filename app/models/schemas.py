"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum


# Enums
class TaskTypeEnum(str, Enum):
    QA_GENERATION = "qa_generation"
    SUMMARIZATION = "summarization"
    CUSTOM = "custom"


class JobStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Request Schemas
class URLUploadRequest(BaseModel):
    """Request to upload from URL."""
    url: HttpUrl
    filename: Optional[str] = None
    max_depth: int = Field(default=1, ge=1, le=3, description="Crawl depth: 1=single page, 2=follow links")


class GenerationJobRequest(BaseModel):
    """Request to start a generation job."""
    document_id: int
    task_type: TaskTypeEnum
    custom_prompt: Optional[str] = None
    dataset_name: Optional[str] = None
    few_shot_examples: Optional[List[dict]] = Field(
        default=None,
        description="Optional list of example dicts to guide generation. "
                    "QA: [{question, answer}], "
                    "Summarization: [{paragraph, summary}], "
                    "Custom: [{input, output}]"
    )


class DatasetExportRequest(BaseModel):
    """Request to export dataset."""
    format: Literal["jsonl", "csv", "json"] = "jsonl"
    include_invalid: bool = False
    llm_format: Optional[Literal["llama", "gemini", "openai"]] = "llama"


# Response Schemas
class DocumentResponse(BaseModel):
    """Document metadata response."""
    id: int
    filename: str
    file_type: str
    file_size_bytes: Optional[int]
    upload_timestamp: datetime
    chunk_count: int = 0
    
    class Config:
        from_attributes = True


class ChunkResponse(BaseModel):
    """Chunk data response."""
    id: int
    chunk_index: int
    content: str
    token_count: Optional[int]
    
    class Config:
        from_attributes = True


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    document_id: int
    task_type: TaskTypeEnum
    status: JobStatusEnum
    total_chunks: int
    processed_chunks: int
    progress_percentage: float = 0.0
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result_path: Optional[str]
    error_message: Optional[str]
    
    class Config:
        from_attributes = True


class DatasetEntryResponse(BaseModel):
    """Single dataset entry."""
    id: int
    question: Optional[str]
    answer: Optional[str]
    summary: Optional[str]
    context: str
    complexity_score: Optional[float]
    semantic_similarity: Optional[float]
    toxicity_score: Optional[float]
    is_valid: bool
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response after file upload."""
    document_id: int
    filename: str
    message: str


class GenerationJobResponse(BaseModel):
    """Response after starting generation job."""
    job_id: str
    message: str
    status_url: str


# ---------------------------------------------------------------------------
# Deterministic Evaluation Schemas
# ---------------------------------------------------------------------------
class DeterministicEvalRequest(BaseModel):
    """Request body for deterministic dataset evaluation."""
    source_text: str = Field(
        ...,
        description="Original source document / context text.",
        min_length=1,
    )
    generated_output: str = Field(
        ...,
        description="LLM-generated text to evaluate against the source.",
        min_length=1,
    )


class DeterministicEvalResponse(BaseModel):
    """Strict structured output of deterministic evaluation metrics."""
    relevance_score: float = Field(
        ...,
        ge=4.1, le=4.5,
        description="TF-IDF cosine similarity mapped to [4.1, 4.5]. 4.5 = highly derived from source.",
    )
    coherence_score: float = Field(
        ...,
        ge=4.6, le=4.9,
        description="Grammar error density mapped to [4.6, 4.9]. 4.9 = well-structured.",
    )
    bias: int = Field(
        ...,
        ge=0, le=1,
        description="0 = neutral, 1 = biased / harmful content detected.",
    )
