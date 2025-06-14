import uvicorn
from fastapi import FastAPI,  HTTPException
from api import predictions, training
from service.prediction_pipeline import init_torch_model
import logging
from logging.handlers import RotatingFileHandler
import os


app = FastAPI()

app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(training.router, prefix="/train", tags=["training"])

model_dict = {}
active_model_id = None


def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, "app.log")

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(logging.INFO)
    fastapi_logger.addHandler(handler)

    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.addHandler(handler)

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.setLevel(logging.INFO)
    uvicorn_error_logger.addHandler(handler)


@app.on_event("startup")
async def startup_event():
    setup_logging()
    logging.info("Starting application...")

    global model_dict
    global active_model_id

    model_id = "pairwise_binary_classifier"
    try:
        model_dict[model_id] = init_torch_model(model_id)
        active_model_id = model_id
        logging.info(f"Model {model_id} successfully loaded with weights.")
    except Exception as e:
        logging.error(f"Failed to load model {model_id}: {str(e)}")
        raise


@app.get("/models")
async def list_models():
    logging.info("Listing available models")
    model_info = {model_id: {
        "parameters": sum(p.numel() for p in model.parameters()),
        "layers": [str(layer) for layer in model.layers]
    } for model_id, model in model_dict.items()}

    return {"models": model_info, "active_model": active_model_id}


# Данная функция необходима на будущее для сравнение моделей, пока же мы работаем с предобученной моделью, автоматически загружаемой при запуске сервера
@app.post("/set/{model_id}")
async def set_active_model(model_id: str):
    logging.info(f"Attempting to set active model to {model_id}")
    global active_model_id
    if model_id in model_dict:
        active_model_id = model_id
        logging.info(f"Active model set to: {active_model_id}")
        return {"message": f"Активная модель установлена: {active_model_id}"}
    else:
        logging.warning(f"Model not found: {model_id}")
        raise HTTPException(status_code=404, detail="Модель не найдена")


@app.get("/")
async def root():
    logging.info("Root endpoint accessed")
    return {"message": "Welcome to the prediction API!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)