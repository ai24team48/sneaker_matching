import uvicorn
from fastapi import FastAPI,  HTTPException
from api import predictions, training
from service.prediction_pipeline import init_torch_model

app = FastAPI()

app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(training.router, prefix="/train", tags=["training"])

model_dict = {}
active_model_id = None


@app.on_event("startup")
async def startup_event():
    global model_dict
    global active_model_id

    model_id = "pairwise_binary_classifier"
    model_dict[model_id] = init_torch_model(model_id)
    active_model_id = model_id
    print("Модель PairwiseBinaryClassifier успешно загружена с весами.")


@app.get("/models")
async def list_models():
    model_info = {model_id: {
        "parameters": sum(p.numel() for p in model.parameters()),
        "layers": [str(layer) for layer in model.layers]
    } for model_id, model in model_dict.items()}

    return {"models": model_info, "active_model": active_model_id}


# Данная функция необходима на будущее для сравнение моделей, пока же мы работаем с предобученной моделью, автоматически загружаемой при запуске сервера
@app.post("/set/{model_id}")
async def set_active_model(model_id: str):
    global active_model_id
    if model_id in model_dict:
        active_model_id = model_id
        return {"message": f"Активная модель установлена: {active_model_id}"}
    else:
        raise HTTPException(status_code=404, detail="Модель не найдена")


@app.get("/")
async def root():
    return {"message": "Welcome to the prediction API!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)