from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ (the parent of app/) — anchor point for .env and default storage_dir so
# behavior doesn't depend on the launching process's current working directory
# (e.g. a dev-server runner that starts uvicorn with --app-dir from a different cwd).
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_provider: str = "openai_compatible"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    # Optional — only some providers/models (e.g. Groq's gpt-oss family) support this.
    # When set, it's forwarded as-is; left empty it's omitted so strictly OpenAI-spec
    # providers that reject unknown fields are unaffected.
    llm_reasoning_effort: str = ""

    max_upload_mb: float = 25
    max_rows: int = 500_000
    storage_dir: str = str(BACKEND_DIR / "storage")
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def storage_path(self) -> Path:
        # Resolve a relative storage_dir against BACKEND_DIR (not the process cwd) so
        # storage location doesn't depend on how/where the server process was launched.
        path = Path(self.storage_dir)
        if not path.is_absolute():
            path = BACKEND_DIR / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
