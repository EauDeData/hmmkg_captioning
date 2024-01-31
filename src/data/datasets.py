import networkx as nx
import pandas as pd
import os
from tqdm import tqdm

from src.data.data_defaults import (NON_VALID_GRAPH_NODE_CATEGORIES, PATH_TO_GRAPH_GEXF, PATH_TO_IMAGES_TSV,
                                    IMAGES_PARENT_FOLDER)

'''
nodes = nx.read_gexf(os.path.join(DEFAULT_OUTPUT_FOLDER, DEFAULT_PATH_TO_MERGED_GRAPH)).nodes(data = True)
for node_id, node in tqdm([(n, data) for n, data in nodes if data['node_type'] in valid_categories], desc='Downloading images...'):

'''

class CaptioningDataset:
    def __init__(self, random_walk_leng: int, neighbor_context_window: int,
                 path_to_gexf_context: str = PATH_TO_GRAPH_GEXF,
                 path_to_images_csv: str = PATH_TO_IMAGES_TSV,
                 images_parent_folder: str = IMAGES_PARENT_FOLDER, split: str = 'test'):

        self.graph = nx.read_gexf(path_to_gexf_context)
        self.ugraph = self.graph.to_undirected()
        self.images = pd.read_csv(path_to_images_csv, sep='\t')
        self.random_walk_leng = random_walk_leng

        self.data_items = []
        for idx, (node_id, path, available, train) in tqdm(enumerate(
                zip(
                    *[self.images[column] for column in ['image_node_title', 'subpath', 'missing', 'train']]
                    )
        ), total=len(self.images)):
            item = {
                'node_id': node_id,
                'data_path': os.path.join(
                    images_parent_folder,
                    path
                ),
                'captions': [data['content'] for data in
                             [self.graph.nodes[x] for x in self.graph.neighbors(node_id)] if data['node_type'] == 'text_content']
                # Navigate the graph to add all related "has caption" edges.
                # Remove then such captions to avoid leaks
            }

            if not available or ('train' if train else 'test') != split: continue
            if neighbor_context_window:

                item['context'] = nx.ego_graph(self.ugraph, node_id, radius=neighbor_context_window)
                leaking_edges = [(src_node, tgt_node) for src_node, tgt_node, data in
                       item['context'].edges(data = True) if data['edge_type'] in ['has_caption', 'in_caption']]

                for edge in leaking_edges:
                    for node in edge:
                        if node in item['context'].nodes and node!=node_id:
                            item['context'].remove_node(node)


            else: item['context']: None
            self.data_items.append(item)
            break

    def __getitem__(self, idx):
        # Aqui agafarem el graph i farem un random walk
        data_item = self.data_items[idx]
        random_walk_sequence = list(nx.generate_random_paths(
            data_item['context'],
            sample_size=1, path_length=self.random_walk_leng, weight=None))[0]
        print(f"Random walk starting from node {data_item['node_id']} with length {self.random_walk_leng}: {random_walk_sequence}")


class EdgePredictionDataset:
    '''

    Aquí si que haurem de pensar com donem les tripletes, però això és un problema per l'Adri
        del futur.
    '''