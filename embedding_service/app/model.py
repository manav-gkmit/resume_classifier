import numpy as np
from app.config import settings
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def load_model() -> None:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBED_MODEL)


def get_model() -> SentenceTransformer:
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


def get_embedding(texts: list[str]) -> np.ndarray:
    model = get_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
    )

    return embeddings
