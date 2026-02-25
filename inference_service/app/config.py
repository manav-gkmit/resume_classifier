import logging

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    EMBED_URL: str

    class Config:
        env_file = ".env"


settings = Settings()

logger = logging.getLogger(__name__)
logger.debug("config: OPENAI_API_KEY set=%s", bool(settings.OPENAI_API_KEY))
logger.debug("config: EMBED_URL=%s", settings.EMBED_URL)
