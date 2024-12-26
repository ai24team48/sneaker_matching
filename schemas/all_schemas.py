from pydantic import BaseModel
from typing import List

class TrainingRequest(BaseModel):
    n_epochs: int
    batch_size: int

class TrainingResponse(BaseModel):
    message: str
    best_val_loss: float

class PredictionResponse(BaseModel):
    variantid1: List[int]
    variantid2: List[int]
    target: List[float]