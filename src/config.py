"""Central configuration loaded from environment / .env file."""
from __future__ import annotations

from pathlib import Path

#from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

#load_dotenv(ROOT / ".env", override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    docs_dir: Path = ROOT / "hospital_documents"
    chroma_dir: Path = ROOT / ".chroma"

    top_k: int = 5
    min_similarity: float = 0.25
    stale_days: int = 730


settings = Settings()
