from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    env: str = "development"
    gemini_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./data/sessions.db"
    model_dir: str = "./models"
    data_dir: str = "./data"
    allowed_origins: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"