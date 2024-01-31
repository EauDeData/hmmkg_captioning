import torch.nn as nn
from typing import Dict, List

class GenericEmbedding(nn.Module):
    def __init__(self, type_to_idx_lut: Dict, embedding_size: int):
        super(GenericEmbedding, self).__init__()

        self.str2idxs = type_to_idx_lut
        self.embedding = nn.Embedding(len(self.str2idxs), embedding_dim=embedding_size)

    def forward(self, batch):
        return self.embedding([self.str2idxs[token] for token in batch['tokens']]) # Todo: This might be sub-efficient

