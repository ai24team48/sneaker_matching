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


def make_predictions(sampled_df):
    text_emb1 = torch.tensor(sampled_df["name_bert_64_1"].tolist(), dtype=torch.float32)
    text_emb2 = torch.tensor(sampled_df["name_bert_64_2"].tolist(), dtype=torch.float32)
    img_emb1 = torch.tensor(sampled_df["main_pic_embeddings_resnet_v1_1"].tolist(), dtype=torch.float32)
    img_emb2 = torch.tensor(sampled_df["main_pic_embeddings_resnet_v1_2"].tolist(), dtype=torch.float32)

    model = init_torch_model()
    model.eval()
    with torch.no_grad():
        outputs = model(text_emb1, img_emb1, text_emb2, img_emb2)
        probas = outputs.squeeze().cpu().numpy()
        predictions = (outputs > 0.5).float().squeeze().cpu().numpy()

    sampled_df["predictions"] = predictions
    sampled_df["probas"] = probas

    return sampled_df


