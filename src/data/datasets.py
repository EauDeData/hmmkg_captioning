import networkx as nx
import pandas as pd
import os

from src.data.data_defaults import (NON_VALID_GRAPH_NODE_CATEGORIES, PATH_TO_GRAPH_GEXF, PATH_TO_IMAGES_TSV,
                                    IMAGES_PARENT_FOLDER)

'''
nodes = nx.read_gexf(os.path.join(DEFAULT_OUTPUT_FOLDER, DEFAULT_PATH_TO_MERGED_GRAPH)).nodes(data = True)
for node_id, node in tqdm([(n, data) for n, data in nodes if data['node_type'] in valid_categories], desc='Downloading images...'):

'''

class CaptioningDataset:
    def __init__(self, neighbor_context_window: int,
                 path_to_gexf_context: str = PATH_TO_GRAPH_GEXF,
                 path_to_images_csv: str = PATH_TO_IMAGES_TSV,
                 images_parent_folder: str = IMAGES_PARENT_FOLDER, split: str = 'train'):

        self.graph = nx.read_gexf(path_to_gexf_context)
        self.images = pd.read_csv(path_to_images_csv, sep='\t')

        data_items = []
        for idx, (node_id, path, available, train) in enumerate(
                zip(
                    *[self.images[column] for column in ['image_node_title', 'subpath', 'missing', 'train']]
                    )
        ):
            item = {
                'data_path': os.path.join(
                    images_parent_folder,
                    path
                ),
                'captions': None # Navigate the graph to add all related "has caption" edges.
                # Remove then such captions to avoid leaks
            }

            if not available or ('train' if train else 'test') != split: continue
            if neighbor_context_window:
                context_neighbors = nx.single_source_shortest_path_length(self.graph, node_id,
                                                                          cutoff=neighbor_context_window)
            else: item['context']: []




class EdgePredictionDataset:
    '''

    Aquí si que haurem de pensar com donem les tripletes, però això és un problema per l'Adri
        del futur.
    '''