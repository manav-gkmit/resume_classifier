from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    class Config:
        env_file = ".env"


settings = Settings()
