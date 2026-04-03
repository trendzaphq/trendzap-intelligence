"""
AI Provider Configuration

Configures the AI/LLM provider (Groq, OpenAI, etc.) using the OpenAI-compatible SDK.
Groq uses the same OpenAI SDK with a different base_url, making it a drop-in replacement.

Usage:
    from trendzap_intelligence.config import get_ai_client, settings

    client = get_ai_client()
    response = client.chat.completions.create(
        model=settings.ai_model,
        messages=[{"role": "user", "content": "Analyze this trend..."}],
    )
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(_env_path)


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # AI Provider
    ai_provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "groq"))

    # Groq Configuration
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )

    # OpenAI Configuration (fallback)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))

    # API Configuration
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # Redis
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )

    @property
    def ai_model(self) -> str:
        """Return the model name for the active AI provider."""
        if self.ai_provider == "groq":
            return self.groq_model
        return self.openai_model

    @property
    def ai_api_key(self) -> str:
        """Return the API key for the active AI provider."""
        if self.ai_provider == "groq":
            return self.groq_api_key
        return self.openai_api_key

    @property
    def ai_base_url(self) -> str | None:
        """Return the base URL for the active AI provider (None for OpenAI default)."""
        if self.ai_provider == "groq":
            return self.groq_base_url
        return None


# Singleton settings instance
settings = Settings()


def get_ai_client() -> Any:
    """
    Create and return an OpenAI-compatible client configured for the active AI provider.

    Groq is OpenAI-compatible, so we use the same `openai.OpenAI` client
    with a different `base_url` pointing to Groq's API.

    Returns:
        openai.OpenAI: Configured client instance

    Raises:
        ValueError: If no API key is configured for the active provider
    """
    from openai import OpenAI

    if not settings.ai_api_key:
        raise ValueError(
            f"No API key configured for AI provider '{settings.ai_provider}'. "
            f"Set {'GROQ_API_KEY' if settings.ai_provider == 'groq' else 'OPENAI_API_KEY'} "
            f"in your .env file."
        )

    client_kwargs: dict[str, Any] = {
        "api_key": settings.ai_api_key,
    }

    if settings.ai_base_url:
        client_kwargs["base_url"] = settings.ai_base_url

    return OpenAI(**client_kwargs)
