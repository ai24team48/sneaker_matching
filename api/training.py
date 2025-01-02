from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from service.training_pipline import train_model_service
from schemas.all_schemas import TrainingResponse, TrainingRequest
import io
import pandas as pd
import concurrent.futures

router = APIRouter()


@router.post("/", response_model=TrainingResponse)
async def train_model(
        model_name: str,
        additional_training: bool,
        n_epochs: int,
        batch_size: int,
        file: UploadFile = File(...)
):
    if not file.filename.endswith('.pkl'):
        raise HTTPException(status_code=400, detail="Неправильный формат файла. Ожидается Pickle (.pkl).")

    try:
        df = pd.read_pickle(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении файла: {str(e)}")

    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(train_model_service, df, model_name, additional_training, n_epochs, batch_size)
            avg_train_loss, avg_test_loss = await run_in_threadpool(future.result, timeout=10)
    except concurrent.futures.TimeoutError:
        raise HTTPException(status_code=408, detail="Время ожидания завершения обучения истекло.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели: {str(e)}")

    return TrainingResponse(
        message="Модель успешно обучена.",
        avg_train_loss=avg_train_loss,
        avg_test_loss=avg_test_loss
    )
