from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ANTHROPIC_API_KEY: str
    CHROMA_DB_PATH: str = "./data/chroma_db"
    SOP_DATA_PATH: str = "./data/sops"
    # Note: fastembed is used instead of sentence-transformers (Python 3.13 compat)
    # Model is fixed to all-MiniLM-L6-v2 via fastembed's registry
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CONFIDENCE_THRESHOLD: float = 0.70
    CLAUDE_MODEL: str = "claude-sonnet-4-6"
    CHROMA_COLLECTION_NAME: str = "sop_library"
    ESCALATION_EMAIL: str = "quality@pharmaorg.com"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
