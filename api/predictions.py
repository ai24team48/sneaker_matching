from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
import os
from dotenv import load_dotenv
from multiprocessing import Process, Queue, Value
from sneaker_matching.service.prediction_pipeline import pipeline_predict
from sneaker_matching.schemas.all_schemas import PredictionResponse

router = APIRouter()
active_processes = Value('i', 0)

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

NUM_CORES = int(os.getenv("NUM_CORES", 4))
MAX_MODELS = int(os.getenv("MAX_MODELS", 2))
MAX_PROCESSES = NUM_CORES - 1


def process_predictions(df, queue):
    results_file, accuracy, f1, roc_auc = pipeline_predict(df)
    queue.put((results_file, accuracy, f1, roc_auc))


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    if active_processes.value >= MAX_PROCESSES:
        return {"error": "Maximum number of processes reached."}

    with active_processes.get_lock():
        active_processes.value += 1

    queue = Queue()
    process = Process(target=process_predictions, args=(df, queue))
    process.start()
    process.join()

    with active_processes.get_lock():
        active_processes.value -= 1

    if not queue.empty():
        results_file, accuracy, f1, roc_auc = queue.get()
        return FileResponse(results_file, media_type='text/csv', filename=results_file)
    else:
        return {"error": "Failed to process the request."}



