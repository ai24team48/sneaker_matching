from pydantic import BaseModel
from typing import List


class TrainingRequest(BaseModel):
    model_name: str
    additional_training: bool
    n_epochs: int
    batch_size: int


class TrainingResponse(BaseModel):
    message: str
    avg_train_loss: float
    avg_test_loss: float


class PredictionResponse(BaseModel):
    variantid1: List[int]
    variantid2: List[int]
    target: List[float]