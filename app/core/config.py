import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://4.247.29.26:3000", "http://127.0.0.1:3000"]

    # Backblaze B2 Configuration
    B2_KEY_ID: str = os.getenv("B2_KEY_ID", "")
    B2_APPLICATION_KEY: str = os.getenv("B2_APPLICATION_KEY", "")
    B2_BUCKET_NAME: str = os.getenv("B2_BUCKET_NAME", "scholar-ai-papers")

    # Academic APIs
    UNPAYWALL_EMAIL: str = os.getenv("UNPAYWALL_EMAIL", "")

    # Google Gemini API
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # FastAPI Configuration
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8001"))

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    # Gap Analysis Configuration
    GAP_ANALYSIS_LIGHT_MODE_MAX_PAPERS: int = int(os.getenv("GAP_ANALYSIS_LIGHT_MODE_MAX_PAPERS", "5"))
    GAP_ANALYSIS_LIGHT_MODE_VALIDATION_THRESHOLD: int = int(os.getenv("GAP_ANALYSIS_LIGHT_MODE_VALIDATION_THRESHOLD", "1"))


settings = Settings()

def get_settings() -> Settings:
    """Get the application settings."""
    return settings
