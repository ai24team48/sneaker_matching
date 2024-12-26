import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from train_classes import VariantPairDataset, PairwiseBinaryClassifier
from typing import Optional


def normalize_vector(vector: Optional[np.ndarray], dim: Optional[int] = None) -> np.ndarray:
    norm = np.sqrt(np.sum(np.square(vector)))
    if norm > 0.001:
        return vector / norm
    else:
        return vector


def evaluate_model(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        threshold: float = 0.5
) -> tuple[list[float], list[float], list[float], float]:
    model.eval()
    eval_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probas = []

    with torch.no_grad():
        for batch in dataloader:
            targets = batch["target"]
            outputs = model(*list(batch.values())[:-1])
            loss = criterion(outputs, targets.view(-1, 1))
            predictions = (outputs > threshold).float()
            eval_loss += loss.item()

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
            all_probas.extend(outputs.squeeze().cpu().numpy().tolist())

    return all_targets, all_predictions, all_probas, eval_loss / len(dataloader)


def train_model(
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        n_epochs: int
) -> tuple[dict[str, torch.Tensor], float]:
    best_model_state = model.state_dict()
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(*list(batch.values())[:-1])
            loss = criterion(outputs, batch["target"].view(-1, 1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        _, _, _, eval_loss = evaluate_model(model, test_dataloader, criterion)
        if eval_loss < best_val_loss:
            best_model_state = model.state_dict()
        print(
            f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {(train_loss / len(train_dataloader)):.4f}, Eval Loss: {eval_loss:.4f}")
    return best_model_state, best_val_loss


def train_model_service(num_epochs: int, batch_size: int) -> tuple[dict[str, torch.Tensor], float]:
    name_embeds_df = pd.read_parquet("data/text_and_bert.parquet")[["variantid", "name_bert_64"]]
    images_embeds_df = pd.read_parquet("data/resnet.parquet")[
        ["variantid", "main_pic_embeddings_resnet_v1", "pic_embeddings_resnet_v1"]
    ]

    embeds_df = name_embeds_df.merge(images_embeds_df, on="variantid")
    embeds_df["name_bert_64"] = embeds_df["name_bert_64"].apply(normalize_vector)
    embeds_df["main_pic_embeddings_resnet_v1"] = embeds_df["main_pic_embeddings_resnet_v1"].apply(
        lambda x: normalize_vector(x[0]) if x is not None else np.zeros(128))
    embeds_df["pic_embeddings_resnet_v1"] = embeds_df["pic_embeddings_resnet_v1"].apply(
        lambda x: normalize_vector(x[0]) if x is not None else np.zeros(128))

    embed_dict = embeds_df.set_index("variantid").to_dict()

    train_df, test_df = train_test_split(pd.read_parquet("data/train.parquet"), test_size=0.2, random_state=42)

    train_dataset = VariantPairDataset(
        train_df[["variantid1", "variantid2"]].values,
        embed_dict["name_bert_64"],
        embed_dict["main_pic_embeddings_resnet_v1"],
        embed_dict["pic_embeddings_resnet_v1"],
        train_df["target"].values
    )

    test_dataset = VariantPairDataset(
        test_df[["variantid1", "variantid2"]].values,
        embed_dict["name_bert_64"],
        embed_dict["main_pic_embeddings_resnet_v1"],
        embed_dict["pic_embeddings_resnet_v1"],
        test_df["target"].values
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = PairwiseBinaryClassifier(text_emb_size=64, img_emb_size=128, hidden_size=512, nlayers=5)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model_state, best_val_loss = train_model(model, train_dataloader, test_dataloader, optimizer, criterion,
                                                  num_epochs)

   # real, preds, probas, loss = evaluate_model(model, test_dataloader, criterion)

    return best_model_state, best_val_loss
