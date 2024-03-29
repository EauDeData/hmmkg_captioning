import random
import io
import networkx as nx
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from multiprocessing import Pool, Manager
from PIL import Image

from src.data.data_defaults import (PATH_TO_GRAPH_GEXF, PATH_TO_IMAGES_TSV,
                                    IMAGES_PARENT_FOLDER, VALID_CATEGORIES, DATASET_SAVE_PATH_FSTRING, MYSELF_TAG,
                                    EMBEDDING_NODE_CATEGORIES, TEXT_PROCESSOR_NODE_CATEGORIES, NON_ADMITED_FILE_FORMATS)
from src._io.ioutils import read_image_any_format


class CaptioningDataset(Dataset):
    def __init__(self, random_walk_leng: int, neighbor_context_window: int,
                 path_to_gexf_context: str = PATH_TO_GRAPH_GEXF,
                 path_to_images_csv: str = PATH_TO_IMAGES_TSV,
                 images_parent_folder: str = IMAGES_PARENT_FOLDER, split: str = 'test',
                 dataset_checkpoint: str = DATASET_SAVE_PATH_FSTRING,
                 text_processor_nodes=TEXT_PROCESSOR_NODE_CATEGORIES, include_quotes: bool = False):
        self.random_walk_leng = random_walk_leng
        self.to_text_nodes = text_processor_nodes
        dataset_checkpoint_name = f'{split}_{neighbor_context_window}_radius_context_include_quotes_{include_quotes}'

        if (not dataset_checkpoint is None) and (os.path.exists(dataset_checkpoint.format(dataset_checkpoint_name))):
            self.data_items = pickle.load(open(dataset_checkpoint.format(dataset_checkpoint_name),
                                               'rb'))
            return

        self.graph = nx.read_gexf(path_to_gexf_context)
        self.ugraph = self.graph.to_undirected()

        filtered_nodes = [node for node in self.ugraph.nodes(data=True)
                          if node[1].get('node_type') in VALID_CATEGORIES]
        if include_quotes:
            text_nodes_not_linked_to_image = [node for node in self.ugraph.nodes(data=True)
                                              if node[1].get('node_type') == 'text_content' and
                                              all(self.ugraph.nodes[neighbor].get('node_type') != 'image_content'
                                                  for neighbor in
                                                  self.ugraph.neighbors(node[0]))]  # This way we avoid
            # captions and get quotes
        else:
            text_nodes_not_linked_to_image = []

        # Create a subgraph with filtered nodes
        self.clean_graph = self.graph.subgraph(dict(filtered_nodes + text_nodes_not_linked_to_image))

        self.images = pd.read_csv(path_to_images_csv, sep='\t')

        self.data_items = []
        for idx, (node_id, path, available, train, url) in tqdm(enumerate(
                zip(
                    *[self.images[column] for column in ['image_node_title', 'subpath', 'missing', 'train', 'url']]
                )
        ), total=len(self.images), desc=f"Getting {split} dataset ready for you!"):

            item = {
                'node_id': node_id,
                'data_path': os.path.join(
                    images_parent_folder,
                    path
                ),
                'url': url,
                'captions': set([data['content'] for data in
                                 [self.graph.nodes[x] for x in self.graph.neighbors(node_id)] if
                                 data['node_type'] == 'text_content'])
                # Navigate the graph to add all related "has caption" edges.
                # Remove then such captions to avoid leaks
            }

            if not available \
                    or ('train' if train else 'test') != split \
                    or path.split('.')[-1] in NON_ADMITED_FILE_FORMATS \
                    or not len(item['captions']):
                continue

            if neighbor_context_window:

                neighbors_with_media = [nx.ego_graph(self.clean_graph, neighbor, radius=neighbor_context_window)
                                        for neighbor in self.ugraph.neighbors(node_id)
                                        if self.ugraph[node_id][neighbor].get('edge_type') in ['has_media', 'in_media']]

                new_graph = nx.DiGraph()  # In order to create a fully conected graph of the "found" image
                for neighbor in self.ugraph.neighbors(node_id):
                    edge_type = self.ugraph[node_id][neighbor].get('edge_type',
                                                                   'default_type')  # Get edge_type, use 'default_type' if not present

                    if edge_type in ['has_media', 'in_media']:
                        new_graph.add_node(MYSELF_TAG, node_type='special', content=MYSELF_TAG)
                        new_graph.add_edge(MYSELF_TAG, neighbor, edge_type=edge_type)

                C_graph = neighbors_with_media[0]
                for contexts in neighbors_with_media[1:] + [new_graph]:
                    C_graph = nx.compose(C_graph, contexts)
                item['context'] = C_graph.to_undirected()
                item['has_media_origin'] = [neighbor
                                            for neighbor in self.ugraph.neighbors(node_id)
                                            if self.ugraph[node_id][neighbor].get('edge_type') in ['has_media',
                                                                                                   'in_media']]


            else:
                item['context']: None
            self.data_items.append(item)

        if not dataset_checkpoint is None:
            pickle.dump(self.data_items, open(dataset_checkpoint.format(
                dataset_checkpoint_name), 'wb'))

    def __len__(self):
        return len(self.data_items)

    @staticmethod
    def get_graph_data_from_path(data_item, random_walk_sequence, to_text_nodes):
        graph_data = {'to_node_emb': {}, 'to_text_emb': {}, 'edges': [], 'total_edges': []}
        '''
        Cada node:
            node_id: {
            'global_index': N,
            'node_type': str,
            'node_content': str
            }

        Cada edge:
            {
            global_index_src: Ni,
            global_index_dst: Nj,
            edge_type: str   
            }
        '''

        added_nodes = []
        for id_loop, (node_src, node_tgt) in enumerate(zip(random_walk_sequence, random_walk_sequence[1:])):

            edge = (node_src, node_tgt)
            if edge in graph_data['total_edges'] or edge[::-1] in graph_data['total_edges']:
                continue
            else:
                graph_data['total_edges'].append(edge)

            if not node_src in added_nodes:
                node_type, content = data_item['context'].nodes[node_src]['node_type'], \
                    data_item['context'].nodes[node_src]['content']
                toadd_dict = graph_data['to_text_emb'] if node_type in to_text_nodes \
                    else graph_data['to_node_emb']

                toadd_dict[node_src] = {
                    'global_idx': len(added_nodes),
                    'node_type': node_type,
                    'content': content
                }
                added_nodes.append(node_src)

            if not node_tgt in added_nodes:
                node_type, content = data_item['context'].nodes[node_tgt]['node_type'], \
                    data_item['context'].nodes[node_tgt]['content']
                toadd_dict = graph_data['to_text_emb'] if node_type in to_text_nodes \
                    else graph_data['to_node_emb']

                toadd_dict[node_tgt] = {
                    'global_idx': len(added_nodes),
                    'node_type': node_type,
                    'content': content
                }
                added_nodes.append(node_tgt)

            all_nodes_v2 = {**graph_data['to_node_emb'], **graph_data['to_text_emb']}
            node_src_data, node_tgt_data = all_nodes_v2[node_src], all_nodes_v2[node_tgt]
            edge_type = data_item['context'][node_src][node_tgt].get('edge_type')
            graph_data['edges'].extend([{
                'global_index_src': node_src_data['global_idx'],
                'global_index_dst': node_tgt_data['global_idx'],
                'edge_type': edge_type
            },
                {
                    'global_index_src': node_tgt_data['global_idx'],
                    'global_index_dst': node_src_data['global_idx'],
                    'edge_type': edge_type
                },
            ])

        nodes = {**graph_data['to_node_emb'], **graph_data['to_text_emb']}
        num_nodes = len(nodes)
        sorted_nodes = sorted(nodes.values(), key=lambda x: nodes[x['content']]['global_idx'])

        adj_matrix = np.eye(num_nodes, num_nodes)

        for src, dst in [(edge['global_index_src'], edge['global_index_dst']) for edge in graph_data['edges']]:
            adj_matrix[src, dst] = 1

        graph_data['adj'] = adj_matrix
        graph_data['listed_nodes'] = sorted_nodes

        return graph_data

    def __getitem__(self, idx):

        # Aqui agafarem el graph i farem un random walk
        data_item = self.data_items[idx]
        random_walk_sequence = list(nx.generate_random_paths(
            data_item['context'],
            sample_size=1, path_length=self.random_walk_leng, weight=None))[0]

        graph_data = self.get_graph_data_from_path(data_item, random_walk_sequence, self.to_text_nodes)
        graph_data['random_walk_original_seq'] = random_walk_sequence

        return {'image': read_image_any_format(data_item['data_path']), 'graph_data': graph_data,
                'caption': random.choice(list(data_item['captions']))}

    def get_context_stats(self, idx):
        data_item = self.data_items[idx]['context']
        node_type_counts = {node_type: sum(1 for node in data_item.nodes(data=True)
                                           if node[1].get('node_type') == node_type)
                            for node_type in VALID_CATEGORIES + ['text_content', 'image_content']}
        return {**node_type_counts, **{'total': len(data_item)}}


class CocoCaption(CaptioningDataset):
    def __init__(self, root, ann, to_text_nodes=[]):
        self.to_text_nodes = to_text_nodes
        self.root = root
        self.annot = [(self._process(val['image_id']), val['caption'])
                      for val in ann['annotations']]

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id)).convert('RGB')
        the_void = nx.Graph()  # Trick the data collator into thinking this is a KG dataset
        the_void.add_node(MYSELF_TAG, node_type='special', content=MYSELF_TAG)
        the_void.add_node('None', node_type='event', content='None')
        the_void.add_edge(MYSELF_TAG, 'None')
        graph_data = self.get_graph_data_from_path({'context': the_void},
                                                   [MYSELF_TAG, 'None'], self.to_text_nodes)

        return {'image': image, 'graph_data': graph_data,
                'caption': caption}


class GCCaptions(CaptioningDataset):
    def __init__(self, root, split='training', to_text_nodes=[]):

        self.to_text_nodes = to_text_nodes
        self.root = root

        df_path = 'Validation_GCC-1.1.0-Validation.tsv' if split == 'validation' else 'Train_GCC-training.tsv'
        df_report_path = 'downloaded_validation_report.tsv' if split == 'validation' else 'downloaded_training_report.tsv'

        df = pd.read_csv(os.path.join(root, df_path), delimiter='\t', names=['caption', 'url'])
        df_report = pd.read_csv(os.path.join(root, df_report_path), delimiter='\t',
                                names=['path', 'split', 'format', 'what', 'err_code', 'url'])

        self.annot = Manager().list()
        self.parallel_process(df, df_report, split)
        self.annot = list(self.annot)

        print(f'File processed with length, {len(self)}')
    def __len__(self):
        return len(self.annot)
    def parallel_process(self, df, df_report, split):
        root = self.root  # Assuming self.root is defined somewhere in your class
        tasks = []
        for row_number, (caption, path, mimetype) in tqdm(
                enumerate(
                    zip(df['caption'], df_report['path'], df_report['format'])),
                total=len(df_report), desc=f"Processing {split} split from GCCaptions"):
            tasks.append((row_number, caption, path, mimetype, root, self.annot))

        # Adapt the number of process to the most convenient
        with Pool(64) as pool:
            pool.map(self.process_image, tasks)
    @staticmethod
    def process_image(args):
        row_number, caption, path, mimetype, root, annot_shared = args
        if path == path and mimetype.lower() in [t.lower() for t in ['image/jpeg', 'image/jpg', 'image/png',
                                                                     'jpg', 'JPEG', 'image/jpeg, image/jpeg']]:
            try:
                _ = Image.open(os.path.join(root, path)).convert('RGB')
                annot_shared.append((path, caption))
            except OSError:
                pass
    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        file = os.path.join(self.root, image_id)
        image = Image.open(file).convert('RGB')
        the_void = nx.Graph()  # Trick the data collator into thinking this is a KG dataset
        the_void.add_node(MYSELF_TAG, node_type='special', content=MYSELF_TAG)
        the_void.add_node('None', node_type='event', content='None')
        the_void.add_edge(MYSELF_TAG, 'None')
        graph_data = self.get_graph_data_from_path({'context': the_void},
                                                   [MYSELF_TAG, 'None'], self.to_text_nodes)

        return {'image': image, 'graph_data': graph_data,
                'caption': caption}


class EdgePredictionDataset:
    '''

    Aquí si que haurem de pensar com donem les tripletes, però això és un problema per l'Adri
        del futur.
    '''
