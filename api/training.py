from fastapi import APIRouter
from ..service.training_pipline import train_model_service
from ..schemas.all_schemas import TrainingResponse, TrainingRequest

router = APIRouter()


@router.post("/", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    best_model_state, best_val_loss = train_model_service(request.n_epochs, request.batch_size)
    return TrainingResponse(
        message="Модель успешно обучена.",
        best_val_loss=best_val_loss
    )
