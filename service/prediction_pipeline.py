import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from train_classes import PairwiseBinaryClassifier, VariantPairDataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def init_torch_model() -> PairwiseBinaryClassifier:
    model = PairwiseBinaryClassifier(text_emb_size=64, img_emb_size=128, hidden_size=512, nlayers=5)
    model.load_state_dict(
        torch.load(
            "models/pairwise_binary_classifier.pth",
            map_location=torch.device('cpu')
        )
    )
    return model


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> tuple[list[float], list[float], list[float], float]:
    model.eval()
    all_targets = []
    all_predictions = []
    all_probas = []

    with torch.no_grad():
        for batch in dataloader:
            text_emb1 = batch['text_emb1']
            img_emb1 = batch['img_emb1']
            text_emb2 = batch['text_emb2']
            img_emb2 = batch['img_emb2']
            targets = batch['target']

            criterion = nn.BCELoss()
            outputs = model(text_emb1, img_emb1, text_emb2, img_emb2)
            loss = criterion(outputs, batch["target"].view(-1, 1))
            predictions = (outputs > 0.5).float()

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(predictions.squeeze().cpu().numpy().tolist())
            all_probas.extend(outputs.squeeze().cpu().numpy().tolist())

    return all_targets, all_predictions, all_probas, loss.item()


def pipeline_predict(sampled_df):
    text_embeddings_sampled = sampled_df["name_bert_64_1"]
    img_embeddings_sampled = sampled_df["main_pic_embeddings_resnet_v1"]
    sampled_variant_pairs = sampled_df[["variantid1", "variantid2"]].values
    targets_sampled = sampled_df["target"].values

    sampled_dataset = VariantPairDataset(
        variant_pairs=sampled_variant_pairs,
        text_embeddings=text_embeddings_sampled,
        img_embeddings=img_embeddings_sampled,
        targets=targets_sampled
    )

    sampled_dataloader = DataLoader(sampled_dataset, batch_size=15, shuffle=False)
    model = init_torch_model()
    real, preds, probas, loss = evaluate_model(model, sampled_dataloader)

    accuracy = accuracy_score(real, preds)
    f1 = f1_score(real, preds)
    roc_auc = roc_auc_score(real, probas)

    results_df = pd.DataFrame({
        'variantid1': sampled_variant_pairs[:, 0],
        'variantid2': sampled_variant_pairs[:, 1],
        'target': real,
        'prediction': preds,
        'probability': probas
    })

    results_file = "predictions_results.csv"
    results_df.to_csv(results_file, index=False)

    return results_file, accuracy, f1, roc_auc


