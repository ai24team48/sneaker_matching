from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from service.training_pipline import train_model_service
from schemas.all_schemas import TrainingResponse, TrainingRequest
import io
import pandas as pd
import concurrent.futures
import logging


router = APIRouter()
logger = logging.getLogger()


@router.post("/", response_model=TrainingResponse)
async def train_model(
        model_name: str,
        additional_training: bool,
        n_epochs: int,
        batch_size: int,
        file: UploadFile = File(...)
):

    logger.info(
        f"Начало обучения модели | "
        f"Модель: {model_name} | "
        f"Доп. обучение: {additional_training} | "
        f"Эпохи: {n_epochs} | "
        f"Размер батча: {batch_size}"
    )

    if not file.filename.endswith('.pkl'):
        error_msg = f"Неправильный формат файла {file.filename}. Ожидается Pickle (.pkl)"
        logger.error(error_msg)
        raise HTTPException(status_code=400, detail="Неправильный формат файла. Ожидается Pickle (.pkl).")

    try:
        logger.info(f"Чтение файла {file.filename}")
        df = pd.read_pickle(io.BytesIO(await file.read()))
        logger.info(f"Успешно прочитано {len(df)} строк")
    except Exception as e:
        error_msg = f"Ошибка чтения файла: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=f"Ошибка при чтении файла: {str(e)}")

    try:
        logger.info("Запуск обучения в ProcessPoolExecutor")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future = executor.submit(train_model_service, df, model_name, additional_training, n_epochs, batch_size)
            avg_train_loss, avg_test_loss = await run_in_threadpool(future.result, timeout=10)

        logger.info(
            f"Обучение завершено | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Test loss: {avg_test_loss:.4f}"
        )
    except concurrent.futures.TimeoutError:
        error_msg = "Таймаут обучения (10s)"
        logger.error(error_msg)
        raise HTTPException(status_code=408, detail="Время ожидания завершения обучения истекло.")
    except Exception as e:
        error_msg = f"Ошибка обучения: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обучении модели: {str(e)}")

    return TrainingResponse(
        message="Модель успешно обучена.",
        avg_train_loss=avg_train_loss,
        avg_test_loss=avg_test_loss
    )
