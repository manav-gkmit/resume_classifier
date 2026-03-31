import logging

import numpy as np
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def get_embeddings(texts: list[str]) -> np.ndarray:
    logger.info("requesting embeddings for %d texts", len(texts))
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{settings.EMBED_URL}/embed",
            json={"texts": texts},
        )
        try:
            response.raise_for_status()
        except Exception:
            logger.exception(
                "embedding service returned error status %s",
                response.status_code,
            )
            raise

    emb_data = response.json().get("embeddings")
    logger.debug("received embeddings with shape %s", np.array(emb_data).shape)
    return np.array(emb_data)
