import io
import asyncio
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
import pandas as pd
import os
from dotenv import load_dotenv
from service.prediction_pipeline import make_predictions
from multiprocessing import Value

router = APIRouter()
active_processes = asyncio.Lock()
active_processes_counter = Value('i', 0)

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

NUM_CORES = int(os.getenv("NUM_CORES", 4))
MAX_MODELS = int(os.getenv("MAX_MODELS", 2))
MAX_PROCESSES = NUM_CORES - 1


async def process_predictions(df: pd.DataFrame):
    try:
        result_df = make_predictions(df)
        return result_df
    except Exception as e:
        print(f"Error processing predictions: {e}")
        return None


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["application/octet-stream", "application/x-pickle"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a pickle file.")

    async with active_processes:
        if active_processes.locked():
            if active_processes_counter.value >= MAX_PROCESSES:
                raise HTTPException(status_code=429, detail="Too many concurrent requests. Please try again later.")

            active_processes_counter.value += 1

            try:
                df = pd.read_pickle(io.BytesIO(await file.read()))
                result_df = await process_predictions(df)

                if result_df is None:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to process the request: {str(result_df)}" if isinstance(result_df, Exception) else "Failed to process the request",
                    )

                csv_file = io.StringIO()
                result_df.to_csv(csv_file, index=False)
                csv_file.seek(0)

                return StreamingResponse(
                    csv_file,
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={file.filename}.csv"},
                )
            finally:
                active_processes_counter.value -= 1
        else:
            raise HTTPException(status_code=500, detail="Failed to acquire the lock.")



