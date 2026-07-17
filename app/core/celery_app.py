"""
Celery application configuration for async task processing.
"""

from celery import Celery
from app.core.config import settings

# Initialize Celery app
celery_app = Celery(
    "llm_data_factory",
    broker=settings.redis_url,
    backend=settings.redis_url
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    imports=[
        'app.services.ingestion.tasks',
        'app.services.generation.tasks'
    ]
)

# Task routing (optional, for advanced setups)
celery_app.conf.task_routes = {
    "app.services.ingestion.tasks.*": {"queue": "celery"},
    "app.services.generation.tasks.*": {"queue": "celery"},
}
