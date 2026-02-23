from app.config import settings
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def load_model() -> None:
    global _model
    _model = SentenceTransformer(settings.EMBED_MODEL)
    return _model


def get_model() -> SentenceTransformer:
    model = load_model()
    if model is None:
        raise RuntimeError("Model not loaded")
    return model


def get_embedding(feature: str):
    model = get_model()

    emb = model.encode(
        sentences=feature,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb
