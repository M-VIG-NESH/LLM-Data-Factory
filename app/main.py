"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import router as api_v1_router
from app.core.database import init_db
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Advanced LLM Training Data Generation Platform",
    version="1.0.0",
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_v1_router)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info(f"{settings.app_name} started successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
