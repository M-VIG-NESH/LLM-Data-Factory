"""
API routes for generation job management.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid
from app.core.database import get_db
from app.core.config import settings
from app.models.schemas import (
    GenerationJobRequest,
    GenerationJobResponse,
    JobStatusResponse,
    DatasetEntryResponse,
    DatasetExportRequest,
    DeterministicEvalRequest,
    DeterministicEvalResponse,
)
from app.models.db_models import GenerationJob, Document, Dataset, JobStatus
from app.services.generation.tasks import generate_dataset_task
from app.services.evaluation.export import export_dataset
from app.services.evaluation.metrics import evaluator
from app.services.evaluation.deterministic_scorer import deterministic_evaluator
from app.services.evaluation.llm_judge import llm_judge
from fastapi.responses import FileResponse
import os
from pydantic import BaseModel

class LLMEvalRequest(BaseModel):
    domain: str
    source_text: str
    generated_output: str

router = APIRouter(prefix="/jobs", tags=["Generation Jobs"])


@router.post("/generate", response_model=GenerationJobResponse)
async def create_generation_job(
    request: GenerationJobRequest,
    db: Session = Depends(get_db)
):
    """
    Start a new generation job.
    """
    # Validate document exists
    document = db.query(Document).filter(Document.id == request.document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job record
    job = GenerationJob(
        job_id=job_id,
        document_id=request.document_id,
        task_type=request.task_type,
        status=JobStatus.PENDING,
        dataset_name=request.dataset_name,
        few_shot_examples=request.few_shot_examples  # Save few-shot examples
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Start async task
    generate_dataset_task.delay(
        job.id,
        request.task_type.value,
        request.custom_prompt,
        request.few_shot_examples  # Pass examples to worker
    )
    
    return GenerationJobResponse(
        job_id=job_id,
        message="Generation job started successfully",
        status_url=f"/api/v1/jobs/{job_id}"
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get status of a generation job.
    """
    job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Calculate progress
    progress = 0.0
    if job.total_chunks > 0:
        progress = (job.processed_chunks / job.total_chunks) * 100
    
    return JobStatusResponse(
        job_id=job.job_id,
        document_id=job.document_id,
        task_type=job.task_type,
        status=job.status,
        total_chunks=job.total_chunks,
        processed_chunks=job.processed_chunks,
        progress_percentage=progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result_path=job.result_path,
        error_message=job.error_message
    )


@router.get("/{job_id}/results", response_model=List[DatasetEntryResponse])
async def get_job_results(
    job_id: str,
    skip: int = 0,
    limit: int = 100,
    include_invalid: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get generated dataset entries for a job.
    """
    job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    query = db.query(Dataset).filter(Dataset.job_id == job.id)
    
    if not include_invalid:
        query = query.filter(Dataset.is_valid == 1)
    
    results = query.offset(skip).limit(limit).all()
    
    return [DatasetEntryResponse.from_orm(entry) for entry in results]


@router.post("/{job_id}/evaluate")
async def evaluate_job_results(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Run quality evaluation on job results.
    """
    job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get all dataset entries
    entries = db.query(Dataset).filter(Dataset.job_id == job.id).all()
    
    evaluated_count = 0
    for entry in entries:
        if entry.question and entry.answer:
            # Evaluate QA pair
            eval_results = evaluator.evaluate_qa_pair(
                entry.question,
                entry.answer,
                entry.context
            )
            
            # Update entry
            entry.complexity_score = eval_results["complexity"]["score"]
            entry.semantic_similarity = eval_results["semantic_similarity"]["score"]
            entry.toxicity_score = eval_results["toxicity"]["score"]
            entry.is_valid = 1 if eval_results["is_valid"] else 0
            
            evaluated_count += 1
    
    db.commit()
    
    return {
        "message": f"Evaluated {evaluated_count} entries",
        "total_entries": len(entries)
    }


@router.post("/{job_id}/export")
async def export_job_results(
    job_id: str,
    request: DatasetExportRequest,
    db: Session = Depends(get_db)
):
    """
    Export job results to file.
    """
    job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get dataset entries
    query = db.query(Dataset).filter(Dataset.job_id == job.id)
    
    if not request.include_invalid:
        query = query.filter(Dataset.is_valid == 1)
    
    entries = query.all()
    
    # Convert to dict
    data = []
    for entry in entries:
        data.append({
            "question": entry.question,
            "answer": entry.answer,
            "summary": entry.summary,
            "context": entry.context,
            "complexity_score": entry.complexity_score,
            "semantic_similarity": entry.semantic_similarity,
            "toxicity_score": entry.toxicity_score,
            "is_valid": bool(entry.is_valid)
        })
    
    # Export
    output_path = export_dataset(
        data,
        format=request.format,
        job_id=job_id
    )
    
    # Update job
    job.result_path = output_path
    db.commit()
    
    return FileResponse(
        output_path,
        media_type="application/octet-stream",
        filename=os.path.basename(output_path)
    )


@router.get("/", response_model=List[JobStatusResponse])
async def list_jobs(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all generation jobs."""
    jobs = db.query(GenerationJob).order_by(
        GenerationJob.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    result = []
    for job in jobs:
        progress = 0.0
        if job.total_chunks > 0:
            progress = (job.processed_chunks / job.total_chunks) * 100
        
        result.append(JobStatusResponse(
            job_id=job.job_id,
            document_id=job.document_id,
            task_type=job.task_type,
            status=job.status,
            total_chunks=job.total_chunks,
            processed_chunks=job.processed_chunks,
            progress_percentage=progress,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            result_path=job.result_path,
            error_message=job.error_message
        ))
    
    return result


# ===========================================================================
# DETERMINISTIC EVALUATION ENDPOINTS (newly added — no existing routes changed)
# ===========================================================================

@router.post("/{job_id}/evaluate-deterministic")
async def evaluate_job_deterministic(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Run the deterministic evaluation (TF-IDF relevance + grammar coherence +
    lexicon bias detection) on all dataset entries within a job.

    For each entry the context (source) is compared against the answer/summary
    (generated output) and scores are returned per entry.
    """
    job = db.query(GenerationJob).filter(GenerationJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    entries = db.query(Dataset).filter(Dataset.job_id == job.id).all()
    if not entries:
        raise HTTPException(status_code=404, detail="No dataset entries found for this job")

    results = []
    for entry in entries:
        # Use answer (QA) or summary as the generated output
        generated = entry.answer or entry.summary or ""
        source    = entry.context or ""

        scores = deterministic_evaluator.evaluate(source, generated)
        results.append({
            "entry_id":        entry.id,
            "relevance_score": scores["relevance_score"],
            "coherence_score": scores["coherence_score"],
            "bias":            scores["bias"],
        })

    # Aggregate summary
    if results:
        avg_relevance = sum(r["relevance_score"] for r in results) / len(results)
        avg_coherence = sum(r["coherence_score"] for r in results) / len(results)
        bias_flagged  = sum(r["bias"] for r in results)
    else:
        avg_relevance = avg_coherence = bias_flagged = 0

    return {
        "job_id":              job_id,
        "total_entries":       len(results),
        "avg_relevance_score": round(avg_relevance, 2),
        "avg_coherence_score": round(avg_coherence, 2),
        "bias_flagged_count":  bias_flagged,
        "per_entry_results":   results,
    }


@router.post("/evaluate", response_model=DeterministicEvalResponse)
async def evaluate_single_pair(request: DeterministicEvalRequest):
    """
    Direct, job-agnostic deterministic evaluation of a single source/generated pair.

    Accepts a source text and a generated output and returns the three metrics
    in strict JSON format — no LLM calls, fully reproducible.

    Example request:
        {"source_text": "The Eiffel Tower is in Paris.",
         "generated_output": "The Eiffel Tower is located in Paris, France."}
    """
    scores = deterministic_evaluator.evaluate(
        request.source_text,
        request.generated_output,
    )
    return DeterministicEvalResponse(**scores)


@router.post("/evaluate-llm")
async def evaluate_single_pair_llm(request: LLMEvalRequest):
    """
    Direct, job-agnostic LLM-as-a-judge evaluation of a single source/generated pair.
    Uses the configured Groq LLM model to provide qualitative grading and critique,
    taking into account the specified data Domain.
    """
    scores = llm_judge.evaluate_pair(
        request.domain,
        request.source_text,
        request.generated_output
    )
    return scores
