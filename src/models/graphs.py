import torch.nn as nn
import torch
from typing import List
from torch_geometric.nn import GATConv, GraphConv, TransformerConv, GAT

from src.models.attentions import ScaledDotProductAttention


class GraphContextGAT(nn.Module):
    def __init__(self, image_processor: nn.Module, image_embedding_size: int,
                 text_processor: nn.Module, text_embedding_size: int, feature_size: int, graph_embedding: nn.Module,
                 depth: int, heads: int, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = .1,
                 device: str='cuda'):

        super(GraphContextGAT, self).__init__()
        self.device = device

        self.text_processor = text_processor
        self.text_projection = nn.Linear(text_embedding_size, feature_size // 2)

        self.image_processor = image_processor # Això podrà ser None quan estem fent caption-only perque
        # mai veurem altres imatges entrenant sols aa captioning, però ho implementem tot per ser consistents
        # quan incorporem el link prediction
        self.image_projection = nn.Linear(image_embedding_size, feature_size)

        self.graph_elements_embedding = graph_embedding # This will contain all the embeddings,
        # the batch already gives you what is what, so all goes through this
        # i.e. from 0-X nodes, x-Y categories... etc
        # A node embedding will be CONCAT([node_features], [category_features]) with small category features
        # Therefore the feature_size = |node_features|+|category_features|
        self.edge_super_projection = nn.Linear(feature_size // 2, feature_size)
        self.feature_size = feature_size

        self.input_to_convolution_projection = torch.nn.Linear(feature_size, in_channels, heads=heads)

        self.depth = depth


        self.gat_backbone = GAT(in_channels, hidden_channels, out_channels, depth)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU()

        self.to(device)

    def forward_graph_data(self, data):

        # This is extremely sub-efficient, please fix this function whenever you have some free minutes to sparse
        batch_size = data['images'].shape[0]

        max_nodes = max(max(x) for x in data['graph']['node_embs']['idxs'] +
                        data['graph']['node_txts']['idxs'] if len(x)
                    )

        nodes_emb = torch.zeros((batch_size, max_nodes + 1, self.feature_size)).to(self.device) # Last node is padding

        for batch_id, (nodes, category, index) in enumerate(zip(data['graph']['node_embs']['tokens'],\
                                                         data['graph']['node_embs']['categories'],\
                                                                data['graph']['node_embs']['idxs'])):

            node_idxs, categories_idxs = torch.tensor(nodes, device=self.device),\
                torch.tensor(category, device=self.device)

            category_embedding = self.graph_elements_embedding(categories_idxs.int())
            node_embedding = self.graph_elements_embedding(node_idxs.int())

            node_features = torch.cat((category_embedding, node_embedding), dim=1)
            nodes_emb[batch_id, index, :] = node_features

        for batch_id, (nodes, category, index) in enumerate(zip(data['graph']['node_txts']['tokens'],\
                                                         data['graph']['node_txts']['categories'],\
                                                                data['graph']['node_txts']['idxs'])):
            # Now for text, extract it with text processor

            categories_idxs = torch.tensor(category, device=self.device) # Sols category,\
            # la frase ja ve ben formatejada del collator

            category_embedding = self.graph_elements_embedding(categories_idxs.int())
            text_nodes = torch.stack(nodes)

            batch_shape = text_nodes.shape[0]
            node_embedding = self.text_processor(text_nodes.view(batch_shape, -1).int())
            projected_text = self.text_projection(node_embedding)

            node_features = torch.cat((category_embedding, projected_text), dim=1)
            nodes_emb[batch_id, index, :] = node_features

        # Lastly, the edges embedding, consider using the padding node (last node in the matrix we've created)
        max_edges = max(len(element) for element in data['graph']['edges']['idxs_coo'] )

        connectivity_matrix = torch.ones((batch_size, 2, max_edges)) * max_nodes + 1
        edge_features = torch.zeros((batch_size, max_edges, self.feature_size))
        print('connectivity:', connectivity_matrix.shape)
        print('edgee features:', edge_features.shape)

        for batch_id, (nodes, category) in enumerate(zip(data['graph']['edges']['idxs_coo'],\
                                                         data['graph']['edges']['categories'])):
            connectivity_matrix[batch_id, :, :len(nodes)] = torch.tensor(nodes).T
            edge_features[batch_id, :len(category), :] = self.edge_super_projection(
                self.graph_elements_embedding(
                    torch.tensor(category)
                )
            )
        edge_projected_features = self.input_to_convolution_projection(self.activation(self.dropout(edge_features)))
        node_projected_features = self.input_to_convolution_projection(self.activation(self.dropout(nodes_emb)))
        return {
            'edge_index': [connectivity_matrix[x].int() for x, edges in
                           zip(range(batch_size), data['graph']['edges']['idxs_coo'])],
            'edge_attr': [edge_projected_features[x] for x, edges in
                           zip(range(batch_size), data['graph']['edges']['idxs_coo'])],
            'node_attr':  [node_projected_features[x] for x in range(batch_size)],
        }
    def forward(self, batch):
        graph_data = self.forward_graph_data(batch)
        data = self.gat_backbone(x=graph_data['node_attr'], edge_attr=graph_data['edge_attr'], edge_index=graph_data['edge_index'],
                                 batch_size=graph_data['batch_size'], batch_vector=None) # TODO: Put batch vector

        image_representation = self.image_projection(self.image_processor(batch['images']))
        print(image_representation.shape)