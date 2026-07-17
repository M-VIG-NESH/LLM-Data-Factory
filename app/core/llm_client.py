"""
Unified LLM client wrapper for Groq and Gemini APIs.
Sequential (synchronous) only — no async.
Provides automatic retry with exponential back-off.
"""

import logging
from typing import Optional, Literal

from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified synchronous interface for multiple LLM providers."""

    def __init__(
        self,
        provider: Literal["groq", "gemini"] = None,
        model: str = None,
    ):
        self.provider = provider or settings.default_llm_provider
        self.model    = model    or settings.default_model
        self._client  = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "groq":
            try:
                from groq import Groq
                self._client = Groq(api_key=settings.groq_api_key)
                logger.info("Initialized Groq client")
            except Exception as e:
                logger.error(f"Failed to initialize Groq: {e}")
                raise

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.gemini_api_key)
                self._client = genai.GenerativeModel(self.model)
                logger.info("Initialized Gemini client")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
    )
    def generate(
        self,
        prompt:      str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate text using the configured LLM (synchronous).

        Retries up to 3 times with exponential back-off (4 s → 8 s → 16 s)
        if Groq is slow or returns a rate-limit error.
        """
        max_tokens  = max_tokens  or settings.max_tokens
        temperature = temperature or settings.temperature

        try:
            if self.provider == "groq":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content

            elif self.provider == "gemini":
                response = self._client.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature":       temperature,
                    },
                )
                return response.text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


# Global client instance
default_llm_client = LLMClient()
