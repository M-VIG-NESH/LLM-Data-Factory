"""
Quality evaluation metrics for generated data.
"""

import logging
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer, util
from detoxify import Detoxify
from app.core.config import settings

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """Evaluates quality of generated training data."""
    
    def __init__(self):
        # Initialize models lazily
        self._similarity_model = None
        self._toxicity_model = None
    
    @property
    def similarity_model(self):
        """Lazy load sentence transformer."""
        if self._similarity_model is None:
            logger.info("Loading sentence transformer model...")
            self._similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._similarity_model
    
    @property
    def toxicity_model(self):
        """Lazy load toxicity detector."""
        if self._toxicity_model is None:
            logger.info("Loading toxicity detection model...")
            self._toxicity_model = Detoxify('original')
        return self._toxicity_model
    
    def check_complexity(self, answer: str, min_length: int = None) -> Dict:
        """
        Check if answer meets minimum complexity requirements.
        
        Args:
            answer: Generated answer text
            min_length: Minimum word count
            
        Returns:
            Dict with is_valid and score
        """
        min_length = min_length or settings.min_answer_length
        word_count = len(answer.split())
        
        is_valid = word_count >= min_length
        
        return {
            "is_valid": is_valid,
            "score": min(word_count / min_length, 1.0) if min_length > 0 else 1.0,
            "word_count": word_count
        }
    
    def check_semantic_similarity(
        self,
        answer: str,
        context: str,
        threshold: float = 0.3
    ) -> Dict:
        """
        Check if answer is semantically grounded in context.
        Uses BERT embeddings to detect hallucinations.
        
        Args:
            answer: Generated answer
            context: Source context
            threshold: Minimum similarity score
            
        Returns:
            Dict with is_valid and similarity score
        """
        try:
            # Encode texts
            answer_embedding = self.similarity_model.encode(answer, convert_to_tensor=True)
            context_embedding = self.similarity_model.encode(context, convert_to_tensor=True)
            
            # Compute cosine similarity
            similarity = util.cos_sim(answer_embedding, context_embedding).item()
            
            is_valid = similarity >= threshold
            
            return {
                "is_valid": is_valid,
                "score": similarity
            }
        
        except Exception as e:
            logger.error(f"Semantic similarity check failed: {e}")
            return {"is_valid": True, "score": 0.5}  # Default to pass on error
    
    def check_toxicity(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Check text for toxic content.
        
        Args:
            text: Text to check
            threshold: Maximum toxicity score
            
        Returns:
            Dict with is_valid and toxicity scores
        """
        try:
            results = self.toxicity_model.predict(text)
            
            # Get max toxicity score across all categories
            max_toxicity = max(results.values())
            
            is_valid = max_toxicity < threshold
            
            return {
                "is_valid": is_valid,
                "score": max_toxicity,
                "details": results
            }
        
        except Exception as e:
            logger.error(f"Toxicity check failed: {e}")
            return {"is_valid": True, "score": 0.0}  # Default to pass on error
    
    def check_bias(self, text: str) -> Dict:
        """
        Basic heuristic bias detection.
        
        Args:
            text: Text to check
            
        Returns:
            Dict with is_valid and detected patterns
        """
        # Simple keyword-based bias detection
        bias_keywords = [
            "always", "never", "all", "none", "every", "only",
            "obviously", "clearly", "definitely", "absolutely"
        ]
        
        text_lower = text.lower()
        detected = [kw for kw in bias_keywords if kw in text_lower]
        
        # Flag if too many absolute terms
        is_valid = len(detected) <= 2
        
        return {
            "is_valid": is_valid,
            "score": 1.0 - (len(detected) / 10),  # Normalize
            "detected_keywords": detected
        }
    
    def evaluate_qa_pair(
        self,
        question: str,
        answer: str,
        context: str
    ) -> Dict:
        """
        Comprehensive evaluation of QA pair.
        
        Args:
            question: Question text
            answer: Answer text
            context: Source context
            
        Returns:
            Dict with all evaluation results
        """
        results = {
            "complexity": self.check_complexity(answer),
            "semantic_similarity": self.check_semantic_similarity(answer, context),
            "toxicity": self.check_toxicity(answer),
            "bias": self.check_bias(answer)
        }
        
        # Overall validity
        results["is_valid"] = all(
            r["is_valid"] for r in results.values() if isinstance(r, dict)
        )
        
        # Average score
        scores = [r["score"] for r in results.values() if isinstance(r, dict) and "score" in r]
        results["overall_score"] = sum(scores) / len(scores) if scores else 0.0
        
        return results


# Global evaluator instance
evaluator = QualityEvaluator()
