import torch.nn as nn
from typing import List
from torch_geometric.nn import GATConv

class GraphContextGAT(nn.Module):
    def __init__(self, image_processor: nn.Module, image_embedding_size: int,
                 text_processor: nn.Module, text_embedding_size: int, feature_size: int,
                 edge_embedding: nn.Module, node_embedding: nn.Module, node_category_embedding: nn.Module,
                 depth: int, heads: int, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = .1,
                 device: str='cuda'):

        super(GraphContextGAT, self).__init__()
        self.device = device

        self.text_processor = text_processor
        self.text_projection = nn.Linear(text_embedding_size, feature_size)

        self.image_processor = image_processor # Això podrà ser None quan estem fent caption-only perque
        # mai veurem altres imatges entrenant sols aa captioning, però ho implementem tot per ser consistents
        # quan incorporem el link prediction
        self.image_projection = nn.Linear(image_embedding_size, feature_size)

        self.edges_emb = edge_embedding # has to return feature_size features
        self.nodes_emb = node_embedding
        self.node_category_embedding = node_category_embedding
        # A node embedding will be CONCAT([node_features], [category_features]) with small category features
        # Therefore the feature_size = |node_features|+|category_features|

        self.depth = depth

        for idx in range(depth):

            # TODO: I don't know if the channels will be a factor to consider
            setattr(self, f"gat_layer_{idx}", GATConv(in_channels if not idx else hidden_channels,
                                                      hidden_channels,
                                                      heads=heads
                                                      )
                    )

        setattr(self, f"gat_layer_{depth}", GATConv(hidden_channels * heads, out_channels, heads=1))

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU()

        self.to(device)
    def forward(self, batch):
        pass # We need to decide what is in the batch
