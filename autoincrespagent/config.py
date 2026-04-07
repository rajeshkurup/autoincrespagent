"""Central settings — all env vars in one place."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Downstream services
    graphserv_url: str = "http://localhost:8080"
    ollama_base_url: str = "http://localhost:11434"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "ir_user"
    mysql_password: str = ""
    mysql_database: str = "incident_response"

    # Agent behaviour
    confidence_threshold: float = 0.75
    max_feedback_iterations: int = 3
    poll_interval_seconds: int = 60


settings = Settings()
