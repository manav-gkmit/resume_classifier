from contextlib import asynccontextmanager

from app.model import get_embedding, get_model, load_model
from app.schema import EmbedRequest
from fastapi import FastAPI, HTTPException

app = FastAPI()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


@app.get("/health")
def health():
    try:
        get_model()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        embedding = get_embedding(req.text)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not ready")

    return {
        "embedding": embedding.tolist(),
        "dimension": int(len(embedding)),
    }
