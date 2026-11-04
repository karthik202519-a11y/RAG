"""Runtime settings, read from environment variables (and the repo-root .env).

Local-first defaults need no external service: SQLite, an on-disk artifact folder,
in-process jobs and an embedded Qdrant. Setting DATABASE_URL / REDIS_URL /
QDRANT_URL / S3_ENDPOINT switches each part to the Docker (scale) profile.
"""
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _secret_key(data_dir: Path) -> str:
    """STUDIO_SECRET_KEY, or a random key persisted in the data folder for local dev."""
    if os.getenv("STUDIO_SECRET_KEY"):
        return os.environ["STUDIO_SECRET_KEY"]
    data_dir.mkdir(parents=True, exist_ok=True)
    f = data_dir / ".secret_key"
    if not f.exists():
        f.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    return f.read_text(encoding="utf-8").strip()


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("STUDIO_DATA_DIR", str(ROOT / "data" / "studio"))))
    database_url: str = ""
    redis_url: str = os.getenv("REDIS_URL", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    job_backend: str = ""  # "thread" or "celery"
    secret_key: str = ""
    token_minutes: int = int(os.getenv("STUDIO_TOKEN_MINUTES", "720"))
    max_upload_mb: int = int(os.getenv("STUDIO_MAX_UPLOAD_MB", "200"))
    job_workers: int = int(os.getenv("STUDIO_JOB_WORKERS", "2"))
    job_max_attempts: int = int(os.getenv("STUDIO_JOB_MAX_ATTEMPTS", "3"))
    cors_origins: str = os.getenv("STUDIO_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
    s3_endpoint: str = os.getenv("S3_ENDPOINT", "")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "")
    s3_bucket: str = os.getenv("S3_BUCKET", "studio-artifacts")
    qdrant_path: str = os.getenv("QDRANT_PATH", "")  # embedded Qdrant folder when QDRANT_URL is unset
    azure_chat_deployment: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_embedding_deployment: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if not self.database_url:
            self.database_url = os.getenv("DATABASE_URL") or f"sqlite:///{(self.data_dir / 'studio.db').as_posix()}"
        if not self.job_backend:
            self.job_backend = os.getenv("STUDIO_JOB_BACKEND") or ("celery" if self.redis_url else "thread")
        if not self.secret_key:
            self.secret_key = _secret_key(self.data_dir)


settings = Settings()
