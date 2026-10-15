"""All settings live here so every other module stays clean."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "google")      # google | anthropic
    llm_model: str = os.getenv("LLM_MODEL", "gemini-flash-latest")
    llm_concurrency: int = int(os.getenv("LLM_CONCURRENCY", "1"))  # free tiers rate-limit hard
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/papers")
    top_k: int = int(os.getenv("TOP_K", "6"))
    chunk_words: int = int(os.getenv("CHUNK_WORDS", "220"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.30"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    max_pages: int = int(os.getenv("MAX_PAGES", "300"))


settings = Settings()
