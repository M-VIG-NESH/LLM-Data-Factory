"""
Celery tasks for generation module.
"""

from celery import Task
from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.db_models import GenerationJob, Chunk, Dataset, JobStatus, TaskType
from app.services.generation.pipeline import GenerationPipeline
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.services.generation.tasks.generate_dataset")
def generate_dataset_task(
    self: Task,
    job_id: int,
    task_type: str,
    custom_prompt: str = None,
    few_shot_examples: list = None
) -> dict:
    """
    Generate training dataset from document chunks.
    
    Args:
        job_id: Database ID of generation job
        task_type: Type of task (qa_generation, summarization, custom)
        custom_prompt: Custom instruction (for custom tasks)
        few_shot_examples: Optional list of example dicts for few-shot prompting
        
    Returns:
        Dict with generation results
    """
    db = SessionLocal()
    
    try:
        # Get job
        job = db.query(GenerationJob).filter(GenerationJob.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Update job status
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Starting generation job {job.job_id} (type: {task_type})")
        if few_shot_examples:
            logger.info(f"Using {len(few_shot_examples)} few-shot examples")
        
        # Get document chunks
        chunks = db.query(Chunk).filter(
            Chunk.document_id == job.document_id
        ).order_by(Chunk.chunk_index).all()
        
        job.total_chunks = len(chunks)
        db.commit()
        
        # Initialize pipeline
        pipeline = GenerationPipeline()
        
        # Process chunks
        for idx, chunk in enumerate(chunks):
            try:
                if task_type == TaskType.QA_GENERATION.value:
                    # Generate QA pairs with optional few-shot examples
                    qa_pairs = pipeline.generate_qa_pairs(
                        chunk.content,
                        few_shot_examples=few_shot_examples
                    )
                    
                    for pair in qa_pairs:
                        dataset_entry = Dataset(
                            job_id=job.id,
                            question=pair["question"],
                            answer=pair["answer"],
                            context=chunk.content
                        )
                        db.add(dataset_entry)
                
                elif task_type == TaskType.SUMMARIZATION.value:
                    # Generate structured Q/C/A summarization records
                    summary_records = pipeline.generate_summary(
                        chunk.content,
                        few_shot_examples=few_shot_examples
                    )

                    if summary_records:
                        for rec in summary_records:
                            dataset_entry = Dataset(
                                job_id=job.id,
                                question=rec.get("question", ""),
                                answer=rec.get("answer", ""),
                                context=rec.get("context", chunk.content),
                                summary=rec.get("answer", ""),  # also store in summary field
                            )
                            db.add(dataset_entry)

                
                elif task_type == TaskType.CUSTOM.value and custom_prompt:
                    # Custom generation with optional few-shot examples
                    output = pipeline.generate_custom(
                        chunk.content,
                        custom_prompt,
                        few_shot_examples=few_shot_examples
                    )
                    
                    if output:
                        dataset_entry = Dataset(
                            job_id=job.id,
                            answer=output,  # Store in answer field
                            context=chunk.content
                        )
                        db.add(dataset_entry)
                
                # Update progress
                job.processed_chunks = idx + 1
                db.commit()
                
                # Update Celery task progress
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': idx + 1,
                        'total': len(chunks),
                        'status': f'Processing chunk {idx + 1}/{len(chunks)}'
                    }
                )
            
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.id}: {e}")
                continue
        
        # Mark job as completed
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        db.commit()
        
        # Count generated entries
        entry_count = db.query(Dataset).filter(Dataset.job_id == job.id).count()
        
        logger.info(f"Job {job.job_id} completed: {entry_count} entries generated")
        
        return {
            "status": "success",
            "job_id": job.job_id,
            "entries_generated": entry_count
        }
    
    except Exception as e:
        logger.error(f"Generation job failed: {e}")
        
        # Mark job as failed
        if job:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.commit()
        
        raise
    
    finally:
        db.close()
