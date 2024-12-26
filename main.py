import uvicorn
from fastapi import FastAPI
from sneaker_matching.api import predictions, training

app = FastAPI()

app.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
app.include_router(training.router, prefix="/train", tags=["training"])

@app.get("/")
async def root():
    return {"message": "Welcome to the prediction API!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)