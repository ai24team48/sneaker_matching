import io
import asyncio
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
import pandas as pd
import os
from dotenv import load_dotenv
from service.prediction_pipeline import make_predictions
from multiprocessing import Value
import logging


router = APIRouter()
logger = logging.getLogger()
active_processes = asyncio.Lock()
active_processes_counter = Value('i', 0)

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

NUM_CORES = int(os.getenv("NUM_CORES", 4))
MAX_MODELS = int(os.getenv("MAX_MODELS", 2))
MAX_PROCESSES = NUM_CORES - 1

logger.info(f"Initialized with NUM_CORES={NUM_CORES}, MAX_MODELS={MAX_MODELS}, MAX_PROCESSES={MAX_PROCESSES}")


async def process_predictions(df: pd.DataFrame):
    try:
        logger.info("Starting predictions processing")
        result_df = make_predictions(df)
        logger.info("Predictions processed successfully")
        return result_df
    except Exception as e:
        logger.error(f"Error processing predictions: {str(e)}", exc_info=True)
        return e


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request for file: {file.filename}")

    if file.content_type not in ["application/octet-stream", "application/x-pickle"]:
        error_msg = f"Unsupported file format: {file.content_type}"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    async with active_processes:
        current_processes = active_processes_counter.value
        logger.debug(f"Active processes: {current_processes}")

        if current_processes >= MAX_PROCESSES:
            error_msg = f"Too many concurrent requests (max {MAX_PROCESSES})"
            logger.warning(error_msg)
            raise HTTPException(status_code=429, detail=error_msg)

        try:
            active_processes_counter.value += 1
            logger.debug(f"Incremented process counter to {active_processes_counter.value}")

            logger.info(f"Reading file {file.filename}")
            df = pd.read_pickle(io.BytesIO(await file.read()))
            logger.info(f"Successfully read DataFrame with shape {df.shape}")

            result_df = await process_predictions(df)

            if result_df is None or isinstance(result_df, Exception):
                error_msg = f"Prediction failed: {str(result_df)}" if result_df else "Unknown error"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

            logger.info("Converting results to CSV")
            csv_file = io.StringIO()
            result_df.to_csv(csv_file, index=False)
            csv_file.seek(0)

            logger.info("Returning prediction results")
            return StreamingResponse(
                csv_file,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={file.filename}.csv"},
            )

        except Exception as e:
            logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            active_processes_counter.value -= 1
            logger.debug(f"Decremented process counter to {active_processes_counter.value}")



