import logging
import re

import joblib
import numpy as np

# each module gets its own logger
logger = logging.getLogger(__name__)

_model = None
_scaler = None
_jd_embeddings = None


def load_classifier():
    global _model, _scaler, _jd_embeddings

    logger.info("loading model and artifacts from disk")
    _model = joblib.load("models/classifier.pkl")
    _scaler = joblib.load("models/scaler.pkl")
    _jd_embeddings = np.load("models/jd_embeddings.npy")
    logger.info("model and artifacts loaded")


def get_jd_embeddings():
    return _jd_embeddings


def predict(features: np.ndarray) -> float:
    logger.debug("predict: received features shape=%s", features.shape)
    scaled = _scaler.transform(features)
    prob = float(_model.predict_proba(scaled)[0, 1])
    logger.debug("predict: probability=%s", prob)
    return prob


def clean_text(text: str) -> str:
    text = text.lower()

    # remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # remove phone numbers (handles +91, (), -, spaces)
    text = re.sub(
        r"\b(\+?\d{1,3}[\s\-]?)?(\(?\d{3}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b", " ", text
    )

    # remove urls
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    text = re.sub(r"[^a-z0-9%\+\-\.\s]", " ", text)

    # Remove extra ------
    text = re.sub(r"-{2,}", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_similarity(resume_embs: np.ndarray) -> np.ndarray:
    jd_embs = get_jd_embeddings()
    logger.debug("computing similarity between resume and jd embeddings")

    sims = [float(np.dot(resume_embs[i], jd_embs[i])) for i in range(2)]

    result = np.array(sims).reshape(1, -1)
    logger.debug("similarity result=%s", result)
    return result
