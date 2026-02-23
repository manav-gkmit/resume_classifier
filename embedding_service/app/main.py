from contextlib import asynccontextmanager

from app.model import get_embedding, get_model, load_model
from app.schema import EmbedRequest
from fastapi import FastAPI, HTTPException


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    try:
        get_model()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/embed")
def embed(req: EmbedRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="Empty input")

    try:
        embeddings = get_embedding(req.texts)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not ready")

    return {
        "embeddings": embeddings.tolist(),
        "dimension": int(embeddings.shape[1]),
    }
