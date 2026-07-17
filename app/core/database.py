"""
Database connection and session management.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for getting database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database and run migrations."""
    # Import all models to ensure they're registered
    from app.models import db_models
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Run migrations
    with engine.connect() as conn:
        # Check if dataset_name column exists in generation_jobs table
        result = conn.execute(text("PRAGMA table_info(generation_jobs)"))
        columns = [row[1] for row in result]
        
        if 'dataset_name' not in columns:
            # Add the column
            conn.execute(text("ALTER TABLE generation_jobs ADD COLUMN dataset_name VARCHAR(255)"))
            conn.commit()
            print("✅ Migration: Added dataset_name column to generation_jobs")
