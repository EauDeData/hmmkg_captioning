import networkx as nx
import json
import os

from src.data.data_defaults import EMBEDDING_NODE_CATEGORIES, GRAPH_TOKENIZER_DEFAULT_PATH, MYSELF_TAG

class GraphTokenizer:
    def __init__(self, graph_path: str, valid_node_categories=EMBEDDING_NODE_CATEGORIES,
                 checkpoint=GRAPH_TOKENIZER_DEFAULT_PATH):

        if os.path.exists(checkpoint):
            self.token_dict = json.load(open(checkpoint, 'r'))
            return

        self.graph = nx.read_gexf(graph)
        self.valid_node_categories = valid_node_categories
        self.token_dict = {}

        # Build the token dictionary
        self.build_token_dict()
        json.dump(self.token_dict, open(checkpoint, 'w'))

    def build_token_dict(self):
        # Tokenize every single node category
        node_categories = set(node_data['node_type'] for node, node_data in self.graph.nodes(data=True))
        node_categories.add('special')

        self.token_dict.update({category: i for i, category in enumerate(node_categories)})

        # Tokenize every node ID if the node_type is in the list of valid categories
        for node, node_data in self.graph.nodes(data=True):
            if node_data['node_type'] in self.valid_node_categories:
                self.token_dict[node] = len(self.token_dict)

        self.token_dict[MYSELF_TAG] = len(self.token_dict)
        # Tokenize every single edge type
        edge_types = set(edge_data['edge_type'] for _, _, edge_data in self.graph.edges(data=True))
        self.token_dict.update({edge_type: len(self.token_dict) for edge_type in edge_types})

    def get_token_dict(self):
        return self.token_dict

