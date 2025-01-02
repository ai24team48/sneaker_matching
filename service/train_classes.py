import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Optional


class VariantPairDataset(Dataset):
    def __init__(
            self,
            variant_pairs: Iterable[tuple[int, int]],
            text_embeddings: dict[int, np.ndarray],
            img_embeddings: dict[int, np.ndarray],
            minor_img_embeddings: dict[int, np.ndarray],
            targets: np.ndarray
    ) -> None:
        self.variant_pairs = variant_pairs
        self.text_embeddings = text_embeddings
        self.img_embeddings = img_embeddings
        self.minor_img_embeddings = minor_img_embeddings
        self.targets = targets

    def __len__(self) -> int:
        return len(self.variant_pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        variantid1, variantid2 = self.variant_pairs[idx]

        text_emb1 = self.text_embeddings[variantid1]
        img_emb1 = np.concatenate((self.img_embeddings[variantid1], self.minor_img_embeddings[variantid1]))
        text_emb2 = self.text_embeddings[variantid2]
        img_emb2 = np.concatenate((self.img_embeddings[variantid2], self.minor_img_embeddings[variantid2]))

        target = self.targets[idx]

        sample = {
            "text_emb1": torch.tensor(text_emb1, dtype=torch.float32),
            "img_emb1": torch.tensor(img_emb1, dtype=torch.float32),
            "text_emb2": torch.tensor(text_emb2, dtype=torch.float32),
            "img_emb2": torch.tensor(img_emb2, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32)
        }

        return sample


class PairwiseBinaryClassifier(nn.Module):
    def __init__(self, text_emb_size: int, img_emb_size: int, hidden_size: int, nlayers: int) -> None:
        super(PairwiseBinaryClassifier, self).__init__()
        input_size = 2 * (text_emb_size + img_emb_size)
        layers = []
        for i in range(nlayers):
            layers.extend(
                [
                    nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.PReLU()
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.scorer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, text_emb1, img_emb1, text_emb2, img_emb2):
        x = torch.cat((text_emb1, img_emb1, text_emb2, img_emb2), dim=-1)
        x = self.layers(x)
        x = self.sigmoid(self.scorer(x))
        return x


class Embedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(Embedding, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class EmbeddingDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            "text_emb1": torch.tensor(row["name_bert_64_1"], dtype=torch.float32),
            "text_emb2": torch.tensor(row["name_bert_64_2"], dtype=torch.float32),
            "img_emb1": torch.tensor(row["main_pic_embeddings_resnet_v1_1"], dtype=torch.float32),
            "img_emb2": torch.tensor(row["main_pic_embeddings_resnet_v1_2"], dtype=torch.float32),
            "target": torch.tensor(row["target"], dtype=torch.float32),
        }

class PairwiseEmbedOrientBinaryClassifier(nn.Module):
    def __init__(
            self,
            text_emb_size: int,
            img_emb_size: int,
            hidden_size: int,
            nlayers: int,
            img_embed_dim: Optional[int] = None
    ) -> None:
        super(PairwiseEmbedOrientBinaryClassifier, self).__init__()
        img_emb_dim = hidden_size // 2 if not img_embed_dim else img_embed_dim
        text_emb_dim = hidden_size - img_emb_dim
        assert text_emb_dim > 0
        self.img_embedder = Embedding(2 * img_emb_size, img_emb_dim)
        self.text_embedder = Embedding(2 * text_emb_size, text_emb_dim)
        layers = []
        for _ in range(nlayers):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.PReLU(),
                    nn.Dropout(0.3)
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.scorer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
            self,
            text_emb1: torch.Tensor,
            img_emb1: torch.Tensor,
            text_emb2: torch.Tensor,
            img_emb2: torch.Tensor
    ) -> torch.Tensor:
        img_emb = self.img_embedder(torch.concat((img_emb1, img_emb2), dim=-1))
        text_emb = self.text_embedder(torch.concat((text_emb1, text_emb2), dim=-1))
        x = torch.concat((img_emb, text_emb), axis=1)
        x = self.layers(x)
        x = self.sigmoid(self.scorer(x))
        return x


class PairwiseItemOrientBinaryClassifier(nn.Module):
    def __init__(
            self,
            text_emb_size: int,
            img_emb_size: int,
            hidden_size: int,
            nlayers: int,
    ) -> None:
        super(PairwiseItemOrientBinaryClassifier, self).__init__()
        v1_embed_dim = hidden_size // 2
        v2_embed_dim = hidden_size - v1_embed_dim
        self.v1_embedder = Embedding(text_emb_size + img_emb_size, v1_embed_dim)
        self.v2_embedder = Embedding(text_emb_size + img_emb_size, v2_embed_dim)
        layers = []
        for _ in range(nlayers):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.PReLU(),
                    nn.Dropout(0.3)
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.scorer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(
            self,
            text_emb1: torch.Tensor,
            img_emb1: torch.Tensor,
            text_emb2: torch.Tensor,
            img_emb2: torch.Tensor
    ) -> torch.Tensor:
        v1_emb = self.v1_embedder(torch.concat((text_emb1, img_emb1), dim=-1))
        v2_emb = self.v2_embedder(torch.concat((text_emb2, img_emb2), dim=-1))
        x = torch.concat((v1_emb, v2_emb), axis=1)
        x = self.layers(x)
        x = self.sigmoid(self.scorer(x))
        return x
