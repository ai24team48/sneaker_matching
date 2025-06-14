import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .train_classes import PairwiseBinaryClassifier, EmbeddingDataset
from datetime import datetime
import os
from dotenv import load_dotenv
import wandb


load_dotenv()


def init_wandb(model_name: str, config: dict):
    try:
        wandb_key = os.getenv('WANDB_API_KEY')
        if wandb_key:
            wandb.login(key=wandb_key)
        else:
            wandb.login()

        run = wandb.init(
            project="pairwise-model-training",
            name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config
        )
        return run
    except Exception as e:
        print(f"W&B initialization failed: {e}")
        return None


def evaluate_model(model, dataloader, criterion, log_to_wandb=False):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            text_emb1 = batch['text_emb1']
            img_emb1 = batch['img_emb1']
            text_emb2 = batch['text_emb2']
            img_emb2 = batch['img_emb2']
            targets = batch['target']

            outputs = model(text_emb1, img_emb1, text_emb2, img_emb2).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predictions = (outputs > 0.5).float()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = (torch.tensor(all_predictions) == torch.tensor(all_targets)).float().mean()

    if log_to_wandb and wandb.run:
        wandb.log({
            "eval/loss": avg_loss,
            "eval/accuracy": accuracy
        })

    return avg_loss, all_targets, all_predictions, accuracy.item()


def train_model(
        model: nn.Module,
        model_name: str,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        use_wandb: bool = True
) -> tuple[float, float]:

    config = {
        "model": model_name,
        "optimizer": type(optimizer).__name__,
        "criterion": type(criterion).__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": train_dataloader.batch_size,
        "num_epochs": num_epochs
    }

    wandb_run = init_wandb(model_name, config) if use_wandb else None

    if wandb_run:
        wandb.watch(model, log="all", log_freq=10)

    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            inputs = {
                'text_emb1': batch['text_emb1'],
                'img_emb1': batch['img_emb1'],
                'text_emb2': batch['text_emb2'],
                'img_emb2': batch['img_emb2']
            }
            targets = batch['target']

            outputs = model(**inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if wandb_run and batch_idx % 10 == 0:
                wandb.log({
                    "batch/train_loss": loss.item(),
                    "batch/learning_rate": optimizer.param_groups[0]['lr']
                })

        avg_train_loss = epoch_train_loss / len(train_dataloader)
        avg_test_loss, _, _, test_accuracy = evaluate_model(
            model, test_dataloader, criterion, log_to_wandb=use_wandb
        )

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"Test Acc: {test_accuracy:.4f}")

        if wandb_run:
            wandb.log({
                "epoch/train_loss": avg_train_loss,
                "epoch/test_loss": avg_test_loss,
                "epoch/test_accuracy": test_accuracy,
                "epoch": epoch + 1
            })

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), f"models/{model_name}_best.pth")
                wandb.save(f"models/{model_name}_best.pth")
                wandb.run.summary["best_test_loss"] = best_test_loss
                wandb.run.summary["best_test_accuracy"] = test_accuracy

    torch.save(model.state_dict(), f"models/{model_name}_final.pth")

    if wandb_run:
        wandb.save(f"models/{model_name}_final.pth")

        model_artifact = wandb.Artifact(
            name=f"model-{model_name}",
            type="model",
            metadata=config
        )
        model_artifact.add_file(f"models/{model_name}_final.pth")
        wandb.log_artifact(model_artifact)

        wandb.finish()

    return avg_train_loss, avg_test_loss


def train_model_service(sampled_df: pd.DataFrame, model_name: str, additional_training: bool, num_epochs: int=10, batch_size: int=64) -> tuple[dict[str, torch.Tensor], float]:

    train_df, test_df = train_test_split(sampled_df, test_size=0.2, random_state=42)

    train_dataset = EmbeddingDataset(train_df)
    test_dataset = EmbeddingDataset(test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PairwiseBinaryClassifier(text_emb_size=64, img_emb_size=128, hidden_size=512, nlayers=5)
    if additional_training:
        model.load_state_dict(torch.load('/models/pairwise_binary_classifier.pth'))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    avg_train_loss, avg_test_loss = train_model(model, model_name, train_dataloader, test_dataloader, optimizer, criterion, num_epochs)
    return avg_train_loss, avg_test_loss
