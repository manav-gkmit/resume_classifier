from pydantic import BaseModel


class EmbedRequest(BaseModel):
    text: list[str]


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimension: int
