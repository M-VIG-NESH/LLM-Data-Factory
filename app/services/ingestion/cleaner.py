"""
Text cleaning and chunking pipeline with semantic deduplication.
"""

import re
import hashlib
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import tiktoken
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class TextCleaner:
    """Deterministic text cleaning using regex rules."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Apply cleaning rules to raw text while preserving paragraph structure.
        """
        # Preserve double newlines (paragraph breaks) by using a placeholder
        text = re.sub(r'\n{2,}', '\n\n', text)   # Normalize multiple blank lines
        
        # Remove page number artifacts
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Fix common encoding issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        
        # Remove special characters but keep punctuation AND newlines
        # Use DOTALL-safe approach: clean per line
        cleaned_lines = []
        for line in text.splitlines():
            line = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\%\#\@]+', '', line)
            line = re.sub(r'[ \t]+', ' ', line).strip()  # Collapse spaces/tabs only (not newlines)
            cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        # Normalize quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        
        return text.strip()


class TextChunker:
    """Smart text chunking with overlap."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize tokenizer for accurate counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("Failed to load tiktoken, using character count approximation")
            self.tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of (chunk_text, token_count) tuples
        """
        chunks = self.splitter.split_text(text)
        result = []
        
        for chunk in chunks:
            token_count = self._count_tokens(chunk)
            result.append((chunk, token_count))
        
        logger.info(f"Created {len(result)} chunks from text")
        return result


class SemanticDeduplicator:
    """Deduplicate chunks using semantic similarity (ChromaDB)."""
    
    def __init__(self, collection_name: str = "document_chunks"):
        # Initialize ChromaDB in-memory
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./data/chroma_db"
        ))
        
        try:
            self.collection = self.client.get_collection(collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def get_chunk_hash(self, text: str) -> str:
        """Generate hash for chunk."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def is_duplicate(self, text: str, similarity_threshold: float = 0.95) -> bool:
        """
        Check if chunk is semantically similar to existing chunks.
        
        Args:
            text: Chunk text
            similarity_threshold: Cosine similarity threshold
            
        Returns:
            True if duplicate found
        """
        try:
            results = self.collection.query(
                query_texts=[text],
                n_results=1
            )
            
            if results['distances'] and results['distances'][0]:
                # ChromaDB returns distance, convert to similarity
                distance = results['distances'][0][0]
                similarity = 1 - distance
                
                if similarity >= similarity_threshold:
                    logger.info(f"Duplicate chunk found (similarity: {similarity:.3f})")
                    return True
            
            return False
        except Exception as e:
            logger.warning(f"Deduplication check failed: {e}")
            return False
    
    def add_chunk(self, text: str, chunk_id: str):
        """Add chunk to deduplication index."""
        try:
            self.collection.add(
                documents=[text],
                ids=[chunk_id]
            )
        except Exception as e:
            logger.error(f"Failed to add chunk to index: {e}")


# Convenience function for full pipeline
def process_text(
    raw_text: str,
    enable_deduplication: bool = True
) -> List[Tuple[str, int, str]]:
    """
    Complete text processing pipeline.
    
    Args:
        raw_text: Raw input text
        enable_deduplication: Whether to deduplicate chunks
        
    Returns:
        List of (chunk_text, token_count, hash) tuples
    """
    # Clean
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean_text(raw_text)
    
    # Chunk
    chunker = TextChunker()
    chunks = chunker.chunk_text(cleaned_text)
    
    # Deduplicate
    if enable_deduplication:
        deduplicator = SemanticDeduplicator()
        result = []
        
        for chunk_text, token_count in chunks:
            chunk_hash = deduplicator.get_chunk_hash(chunk_text)
            
            if not deduplicator.is_duplicate(chunk_text):
                deduplicator.add_chunk(chunk_text, chunk_hash)
                result.append((chunk_text, token_count, chunk_hash))
        
        logger.info(f"Deduplication: {len(chunks)} -> {len(result)} chunks")
        return result
    else:
        # Just add hashes
        return [
            (text, count, hashlib.sha256(text.encode()).hexdigest())
            for text, count in chunks
        ]
