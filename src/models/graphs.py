import torch.nn as nn
import torch
import torch_geometric

from typing import List
from torch_geometric.nn import GATConv, GraphConv, TransformerConv, GAT

from src.models.attentions import ScaledDotProductAttention


class GraphContextGAT(nn.Module):
    def __init__(self, image_processor: nn.Module, image_embedding_size: int,
                 text_processor: nn.Module, text_embedding_size: int, feature_size: int, graph_embedding: nn.Module,
                 depth: int, heads: int, in_channels: int, hidden_channels: int,
                 dropout: float = .1,
                 device: str = 'cuda', freeze_encoder=False):

        super(GraphContextGAT, self).__init__()
        self.device = device

        self.image_processor = image_processor
        self.text_processor = text_processor

        self.text_projection = nn.Linear(text_embedding_size, feature_size)

        self.graph_elements_embedding = graph_embedding
        self.feature_size = feature_size
        self.depth = depth

        # FOR LOOP WITH EVERY LAYER
        gat_feature = 2 * feature_size
        self.gat_backbone = GAT(gat_feature, hidden_channels, depth, feature_size, dropout=dropout, heads=heads)

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=in_channels, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.image_projection = nn.Linear(image_embedding_size, feature_size)

        # We don't need half the token dedicated to node type
        self.downscale_node_types = nn.Linear(feature_size, feature_size // 2)
        self.downscale_node_text = nn.Linear(feature_size, feature_size // 2)

        # Edges are not concatenated with anything else, so they need to do this "fake" concatenation
        self.upscale_edge_features = nn.Linear(feature_size, gat_feature)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU()

        self.to(device)
        if freeze_encoder:
            print("(script) Freezing encoder layers!")
            for param in self.image_processor.parameters():
                param.requires_grad = False

    def forward(self, batch, max_tokens_text_vectors=25):

        node_attributes = self.graph_elements_embedding(batch['graph']['nodes'].to(self.device))
        node_type_features = self.downscale_node_types(
            self.graph_elements_embedding(batch['graph']['node_types'].to(self.device))
        )

        node_attributes_with_types = torch.cat((node_attributes, node_type_features), dim=1)
        edge_attributes = self.upscale_edge_features(
            self.graph_elements_embedding(batch['graph']['edge_types'].to(self.device))
        )

        text_vectors = self.text_projection(self.activation(
            self.text_processor(batch['graph']['text_nodes'][:, :max_tokens_text_vectors].to(self.device))
        ))
        downsampled_text_vectors = self.downscale_node_text(self.activation(text_vectors))

        node_attributes_text_and_type = torch.cat((node_attributes_with_types, downsampled_text_vectors), dim=1)

        processed_graph_data = self.gat_backbone(x=node_attributes_text_and_type,
                                                 edge_attr=edge_attributes,
                                                 edge_index=torch.tensor(batch['graph']['edges'],
                                                                         device=self.device,
                                                                         dtype=torch.int64).T)

        image_tokens = self.image_projection(self.activation(
            self.image_processor(batch['images'].to(self.device))
        ))  # (SEQ, Batch, FEATURE_SIZE)

        node_tokens = processed_graph_data[batch['graph']['batched_nodes']]  # (Batch, SEQ, FEATURE_SIZE)

        # Transpose node_tokens to have the same shape as image_tokens for concatenation
        node_tokens = node_tokens.transpose(0, 1)  # Shape: (SEQ, Batch, FEATURE_SIZE)
        final_vector = torch.cat((image_tokens, node_tokens), dim=0)  # Shape: (2*Seq, Batch, FEATURE_SIZE)

        SEQ, IMTOKENS = final_vector.shape[0], image_tokens.shape[0]

        mask = torch.zeros((SEQ, SEQ), device=self.device)  # Create an all-zero mask initially
        mask[:IMTOKENS] = float('-inf')  # If I'm not mistaken it means the nodes sending messages to the images
        mask[IMTOKENS:, IMTOKENS:] = float('-inf')  # In only one direction
        mask = mask.masked_fill(torch.eye(mask.size(-1), dtype=torch.bool, device=self.device), 0)

        propagated = self.transformer_encoder(final_vector, mask=mask)[:IMTOKENS]  # Taske only contextualized image features

        return {
            'encoder_features': propagated.transpose(0, 1),  # Shape: (Batch, Seq, FEATURE_SIZE)
            'graph_features': node_tokens.transpose(0, 1),
            'image_features': image_tokens.transpose(0, 1),
            'memory_mask': None
        }


class GraphContextTransformerEncoder(nn.Module):
    def __init__(self, image_processor: nn.Module, image_embedding_size: int,
                 text_processor: nn.Module, text_embedding_size: int, feature_size: int,
                 depth: int, heads: int,
                 dropout: float = .1,
                 device: str = 'cuda', freeze_encoder=False, train_text=False, train_vision=False, use_sbert=False):

        super(GraphContextTransformerEncoder, self).__init__()
        self.device = device

        if not use_sbert:
            self.text_processor = text_processor
        else:
            self.text_processor = None
        self.text_projection = nn.Linear(text_embedding_size if not use_sbert else 384, feature_size)

        self.image_processor = image_processor
        self.image_projection = nn.Linear(image_embedding_size, feature_size)

        self.use_sbert = use_sbert

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU()

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4,
                                                   batch_first=True)  # Heads are hardcoded!!
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to(device)

        if freeze_encoder:
            print("(script) Freezing encoder layers!")
            if not train_text and not (self.text_processor is None):
                for param in self.text_processor.parameters():
                    param.requires_grad = False
            if not train_vision:
                for param in self.image_processor.parameters():
                    param.requires_grad = False

    def forward(self, X, node_max_seq_size=7, max_output_tokens=-1):

        image_vector = self.activation(self.image_processor(X['images'].to(self.device))).permute(1, 0, 2)
        projected_image_token = self.dropout(self.image_projection(image_vector))
        if len(projected_image_token) == 2:  # Single feature:
            projected_image_token = projected_image_token.unsqueeze(0)

        if not self.use_sbert:
            nodes = X['nodes'][:, :, :node_max_seq_size].to(self.device)
            node_batch, node_seq, node_leng = nodes.shape
            text_vector = self.activation(self.text_processor(nodes.view(node_batch * node_seq, node_leng))). \
                view(node_batch, node_seq, -1)  # Go back to sequence of nodes

        else:
            text_vector = X['sbert_nodes'].to(self.device)

        projected_node_tokens = self.text_projection(text_vector) * X['nodes_mask'].to(self.device)[:, :, None]
        concat_features = torch.cat((projected_node_tokens, projected_image_token), dim=1)

        out_features = self.transformer_encoder(concat_features)

        return {
            'features': out_features[:, :max_output_tokens],
            'memory_mask': None
        }
