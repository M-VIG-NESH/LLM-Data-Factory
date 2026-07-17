"""Generation module initialization."""

from .prompts import get_qa_prompt, get_summary_prompt, get_custom_prompt
from .pipeline import GenerationPipeline

__all__ = [
    "get_qa_prompt",
    "get_summary_prompt",
    "get_custom_prompt",
    "GenerationPipeline"
]
