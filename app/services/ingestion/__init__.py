"""Ingestion module initialization."""

from .loader import DocumentLoader
from .cleaner import TextCleaner, TextChunker, SemanticDeduplicator, process_text

__all__ = [
    "DocumentLoader",
    "TextCleaner",
    "TextChunker",
    "SemanticDeduplicator",
    "process_text"
]
