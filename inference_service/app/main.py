import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from app.helpers import load_classifier, predict
from app.logger import configure_logging, logger
from app.preprocessing import preprocess
from app.schema import PredictResponse
from fastapi import FastAPI, File, HTTPException, UploadFile

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("lifespan: loading classifier during startup")
    load_classifier()
    logger.info("lifespan: classifier loaded")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=PredictResponse)
async def predict_resume(file: UploadFile = File(...)):
    logger.info("Received /predict request: filename=%s", file.filename)

    if not file.filename.endswith((".pdf", ".docx", ".PDF")):
        logger.warning("Unsupported file type for %s", file.filename)
        raise HTTPException(status_code=415, detail="Only PDF or DOCX allowed")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
        logger.debug("Wrote upload to temporary file %s", tmp_path)

    try:
        features = await preprocess(tmp_path)
        probability = predict(features)
        logger.info("Prediction succeeded probability=%s", probability)

    except Exception:
        logger.exception("Prediction failed for file %s", file.filename)
        raise HTTPException(status_code=500, detail="Prediction failed")

    finally:
        try:
            os.remove(tmp_path)
            logger.debug("Removed temporary file %s", tmp_path)
        except Exception:
            logger.warning("Failed to remove temporary file %s", tmp_path)

    return PredictResponse(probability=probability)
