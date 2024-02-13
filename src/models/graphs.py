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
                 device: str='cuda', freeze_encoder = False):

        super(GraphContextGAT, self).__init__()
        self.device = device

        self.text_processor = text_processor
        self.text_projection = nn.Linear(text_embedding_size, feature_size // 2)

        self.image_processor = image_processor # Això podrà ser None quan estem fent caption-only perque
        # mai veurem altres imatges entrenant sols aa captioning, però ho implementem tot per ser consistents
        # quan incorporem el link prediction
        self.image_projection = nn.Linear(image_embedding_size + feature_size//2, feature_size)
        self.image_token = nn.Parameter(data=torch.rand(1, feature_size // 2))

        self.graph_elements_embedding = graph_embedding # This will contain all the embeddings,
        # the batch already gives you what is what, so all goes through this
        # i.e. from 0-X nodes, x-Y categories... etc
        # A node embedding will be CONCAT([node_features], [category_features]) with small category features
        # Therefore the feature_size = |node_features|+|category_features|
        self.edge_super_projection = nn.Linear(feature_size // 2, feature_size)
        self.feature_size = feature_size

        self.input_to_convolution_projection = torch.nn.Linear(feature_size, in_channels)

        self.depth = depth

        # FOR LOOP WITH EVERY LAYER
        self.gat_backbone = GAT(in_channels, hidden_channels, depth, feature_size, dropout=dropout, heads=heads)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU()

        self.to(device)
        if freeze_encoder:
            print("(script) Freezing encoder layers!")
            for param in self.text_processor.parameters():
                param.requires_grad = False

            for param in self.image_processor.parameters():
                param.requires_grad = False

    def forward_graph_data(self, data):

        batch_size = data['images'].shape[0]

        max_nodes_per_batch_id = [
                            len(text_nodes + emb_nodes) for text_nodes, emb_nodes in\
                                  zip(data['graph']['node_txts']['idxs'], data['graph']['node_embs']['idxs'])
                                  ]

        padding_idx = total_nodes = sum(max_nodes_per_batch_id)
        nodes_emb = torch.zeros((total_nodes + 1, self.feature_size)).to(self.device) # Last node is padding

        per_batch_node_indices = [[] for _ in range(batch_size)]

        for batch_id, (nodes, category, index) in enumerate(zip(data['graph']['node_embs']['tokens'],\
                                                         data['graph']['node_embs']['categories'],\
                                                                data['graph']['node_embs']['idxs'])):

            if not len(nodes): continue
            node_idxs, categories_idxs = torch.tensor(nodes, device=self.device),\
                torch.tensor(category, device=self.device)


            category_embedding = self.graph_elements_embedding(categories_idxs.int())
            node_embedding = self.graph_elements_embedding(node_idxs.int())

            node_features = torch.cat((category_embedding, node_embedding), dim=1)

            offset_indices = [sum(max_nodes_per_batch_id[:batch_id]) + x for x in index]
            nodes_emb[offset_indices, :] = node_features
            per_batch_node_indices[batch_id].extend(offset_indices)

        for batch_id, (nodes, category, index) in enumerate(zip(data['graph']['node_txts']['tokens'],\
                                                         data['graph']['node_txts']['categories'],\
                                                                data['graph']['node_txts']['idxs'])):
            # Now for text, extract it with text processor
            if not len(nodes): continue

            categories_idxs = torch.tensor(category, device=self.device) # Sols category,\
            # la frase ja ve ben formatejada del collator

            category_embedding = self.graph_elements_embedding(categories_idxs.int())
            text_nodes = torch.stack(nodes).to(self.device)

            batch_shape = text_nodes.shape[0]
            node_embedding = self.text_processor(text_nodes.view(batch_shape, -1).int())
            projected_text = self.text_projection(node_embedding)

            node_features = torch.cat((category_embedding, projected_text), dim=1)

            offset_indices = [sum(max_nodes_per_batch_id[:batch_id]) + x for x in index]
            nodes_emb[offset_indices, :] = node_features
            per_batch_node_indices[batch_id].extend(offset_indices)

        # Lastly, the edges embedding, consider using the padding node (last node in the matrix we've created)
        num_edges = sum(len(element) for element in data['graph']['edges']['idxs_coo'])

        connectivity_matrix = torch.zeros((2, num_edges)) - 1
        edge_features = torch.zeros((num_edges, self.feature_size), device=self.device)

        num_features_added=0
        for batch_id, (nodes, category) in enumerate(zip(data['graph']['edges']['idxs_coo'],\
                                                         data['graph']['edges']['categories'])):


            connectivity_matrix[:, num_features_added:num_features_added+len(nodes)] = torch.tensor(nodes).T +\
                                                                 sum(max_nodes_per_batch_id[:batch_id])

            edge_features[num_features_added:num_features_added+len(category), :] = self.edge_super_projection(
                self.graph_elements_embedding(
                    torch.tensor(category).to(self.device)
                )
            )
            num_features_added += len(category)

        edge_projected_features = self.input_to_convolution_projection(self.activation(self.dropout(edge_features)))
        node_projected_features = self.input_to_convolution_projection(self.activation(self.dropout(nodes_emb)))
        max_number_of_nodes = max(len(x) for x in per_batch_node_indices)

        return {
            'edge_index': connectivity_matrix.to(torch.int64).to(self.device),
            'edge_attr': edge_projected_features,
            'node_attr':  node_projected_features,
            'batch_size': batch_size,
            'node_features_batched_offsets': [inds + [padding_idx] * (max_number_of_nodes - len(inds)) for inds
                                              in per_batch_node_indices]

        }
    def forward(self, batch):
        graph_data = self.forward_graph_data(batch)

        data = self.gat_backbone(x=graph_data['node_attr'],
                                 edge_attr=graph_data['edge_attr'],
                                 edge_index=graph_data['edge_index'])

        image_embedding = self.image_processor(batch['images'].to(self.device))
        concatenated_tensor = torch.cat((self.image_token.expand(image_embedding.shape[0], -1),
                                         image_embedding), dim=1)
        image_representation = self.image_projection(concatenated_tensor).unsqueeze(1)

        sequences = torch.nn.functional.embedding(torch.tensor(graph_data['node_features_batched_offsets'],
                                                               device=self.device), data)
        features = torch.cat((sequences, image_representation), dim=1)

        return {'features': features}


class GraphContextTransformerEncoder(nn.Module):
    def __init__(self, image_processor: nn.Module, image_embedding_size: int,
                 text_processor: nn.Module, text_embedding_size: int, feature_size: int,
                 depth: int, heads: int,
                 dropout: float = .1,
                 device: str = 'cuda', freeze_encoder = False):

        super(GraphContextTransformerEncoder, self).__init__()
        self.device = device

        self.text_processor = text_processor
        self.text_projection = nn.Linear(text_embedding_size, feature_size)

        self.image_processor = image_processor
        self.image_projection = nn.Linear(image_embedding_size, feature_size)

        encoder_layers = torch.nn.TransformerEncoderLayer(feature_size, heads, feature_size, dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, depth)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU()

        self.to(device)
        if freeze_encoder:
            print("(script) Freezing encoder layers!")
            for param in self.text_processor.parameters():
                param.requires_grad = False

            for param in self.image_processor.parameters():
                param.requires_grad = False

    def forward(self, X):

        image_vector = self.activation(self.image_processor(X['images'].to(self.device)))
        projected_image_token = self.image_projection(image_vector)[None, :, :]

        batch, seq_size, n_tokens = X['nodes'].shape
        text_vector = self.activation(self.text_processor(X['nodes'].reshape(batch * seq_size, n_tokens).\
                                                          to(self.device)).reshape(batch, seq_size, -1)).\
            transpose(1, 0)

        projected_node_tokens = self.text_projection(text_vector)

        # IMPORTANT!! IMAGE NODE MUST BE THE LAST ONE OF THE SEQUENCE TO MATCH THE MASK
        full_sequence = (torch.cat((projected_node_tokens, projected_image_token), dim = 0).\
                         transpose(1, 0))
        encoded_features = self.transformer_encoder(full_sequence, mask=X['adj_matrix'].to(self.device))
        print(encoded_features.shape)
        exit()
