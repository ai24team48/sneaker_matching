import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
from .train_classes import PairwiseBinaryClassifier, EmbeddingDataset


def evaluate_model(model, dataloader, criterion):
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
    return avg_loss, all_targets, all_predictions


def train_model(
        model: nn.Module,
        model_name: str,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int
) -> tuple[dict[str, torch.Tensor], float]:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            text_emb1 = batch['text_emb1']
            img_emb1 = batch['img_emb1']
            text_emb2 = batch['text_emb2']
            img_emb2 = batch['img_emb2']
            targets = batch['target']

            outputs = model(text_emb1, img_emb1, text_emb2, img_emb2).squeeze()
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)
        avg_test_loss, _, _ = evaluate_model(model, test_dataloader, criterion)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    torch.save(model.state_dict(), f'models/{model_name}.pth')
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
