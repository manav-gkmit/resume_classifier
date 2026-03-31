from pydantic import BaseModel


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimension: int
