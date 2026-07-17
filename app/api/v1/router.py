"""
API v1 router aggregator.
"""

from fastapi import APIRouter
from app.api.v1 import ingestion, jobs

router = APIRouter(prefix="/api/v1")

# Include sub-routers
router.include_router(ingestion.router)
router.include_router(jobs.router)
