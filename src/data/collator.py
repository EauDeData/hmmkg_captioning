import networkx as nx
import torch
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from src.data.data_defaults import MYSELF_TAG, PADDING_TAG
class Collator:
    '''

    A set of collator functinos so we can select them according to the needed approach

    '''

    def __init__(self, transforms, text_tokenizer, graph_tokenizer, padding_token = 0, use_sbert=True):
        self.transforms = transforms
        self.tokenizer = text_tokenizer
        self.graph_tokenizer = graph_tokenizer
        self.padding_token = 0
        self.use_sbert = use_sbert
        if use_sbert:
            self.sbert = SentenceTransformer('all-MiniLM-L12-v2')
            self.sbert.to('cpu')

        return

    @staticmethod
    def pad_sequence_collator(tensor_list, batch_size, max_seq_size, padding_id, token_leng):
        padded_sequence = pad_sequence([torch.cat([t, torch.ones(max_seq_size - len(t), 1, token_leng) * padding_id])
                                        for t in tensor_list],
                                       batch_first=True)

        # Reshape to (SEQ_SIZE, BATCH_SIZE, 30)
        return padded_sequence.view(batch_size, max_seq_size, token_leng).permute(1, 0, 2)

    def simple_encoder_with_adj_collate(self, batch):

        images = torch.stack([self.transforms(sample['image']) for sample in batch])
        max_nodes = max(max(*sample['graph_data']['adj'].shape) for sample in batch)

        all_adjs = torch.zeros(len(batch), max_nodes + 1, max_nodes + 1)
        all_nodes = []
        padding_nodes_mask = torch.ones(len(batch), max_nodes, dtype=torch.float32)
        sbert_embs = []

        for num, sample in enumerate(batch):
            connectivity_matrix = torch.from_numpy(sample['graph_data']['adj'])
            all_adjs[num, :connectivity_matrix.shape[0], :connectivity_matrix.shape[1]] = connectivity_matrix
            nodes = sample['graph_data']['listed_nodes']

            all_adjs[num, -1, :len(nodes)] = 1
            all_adjs[num, :len(nodes), -1] = 1 # Everything connected to the image but the padding

            nodes_in_caption_no_pad = [node['content'] for node in nodes]
            nodes_in_caption = nodes_in_caption_no_pad + ['[PADDING]'] * (max_nodes - len(nodes))
            padding_nodes_mask[num, len(nodes_in_caption_no_pad):] = 0

            padded_nodes = self.tokenizer.tokenize(
                nodes_in_caption
            )

            all_nodes.append(padded_nodes)
            if self.use_sbert:
                sbert_embs.extend([torch.from_numpy(v) for v in self.sbert.encode(nodes_in_caption)])


        stacked_nodes = torch.stack(all_nodes)

        data = {
            'adj_matrix': all_adjs,
            'images': images,
            'nodes': stacked_nodes,
            'captions': self.tokenizer.tokenize([sample['caption'] for sample in batch]).view\
            (images.shape[0], -1),
            'nodes_mask': padding_nodes_mask,
            'sbert_nodes': [] if not len(sbert_embs) else torch.stack(sbert_embs)

        }
        padding_captions = torch.zeros(data['captions'].shape[0], data['captions'].shape[-1],
                                       dtype=torch.float32)
        for num, caption in enumerate(data['captions']):

            padding_start_index = caption.tolist().index(self.tokenizer.eos_token_id) + 1 # Saltar-se  EOS
            padding_captions[num, padding_start_index:] = float('-inf') # -inf ignora, 0 deixa passar

        return {**data, **{'captions_padding': padding_captions}}

    def base_collate_captioning(self, batch):

        ## IMAGE AND CAPTION COLLATED
        images = torch.stack([self.transforms(sample['image']) for sample in batch])

        tokenized_captions = torch.stack([self.tokenizer.tokenize([sample['caption']]) for sample in batch]).view\
            (images.shape[0], -1) # (BATCH_SIZE, context_length)

        padding_captions = torch.zeros(tokenized_captions.shape[0], tokenized_captions.shape[-1],
                                       dtype=torch.float32)

        graphs = [g['graph_data'] for g in batch]
        merged_graph = nx.compose_all(graphs)

        merged_nodes = list(merged_graph.nodes())
        merged_node_types = [merged_graph.nodes[node]['node_type'] for node in merged_nodes]
        merged_edges = ([(merged_nodes.index(edge[0]), merged_nodes.index(edge[1])) for edge in merged_graph.edges()] +
                        [(merged_nodes.index(edge[1]), merged_nodes.index(edge[0])) for edge in merged_graph.edges()])

        merged_edge_types = [merged_graph.edges[edge]['edge_type'] for edge in merged_graph.edges()] * 2

        per_image_nodes = [[] for _ in batch]



        for num, (caption, graph) in enumerate(zip(tokenized_captions,graphs)):

            padding_start_index = caption.tolist().index(self.tokenizer.eos_token_id) + 1 # Saltar-se  EOS
            padding_captions[num, padding_start_index:] = float('-inf') # -inf ignora, 0 deixa passar

            nodes_in_this_graph = [merged_nodes.index(node) for node in graph.nodes()]
            per_image_nodes[num] = nodes_in_this_graph

        tokenized_nodes = torch.tensor([self.graph_tokenizer.token_dict[node] for node in merged_nodes + [PADDING_TAG]],
                                       dtype=torch.int64)
        tokenized_node_types = torch.tensor([self.graph_tokenizer.token_dict[node] for node in merged_node_types
                                             + ['special']],
                                            dtype=torch.int64)
        tokenized_edge_types = torch.tensor([self.graph_tokenizer.token_dict[edge] for edge in merged_edge_types],
                                            dtype=torch.int64)

        max_num_nodes = max([len(x) for x in per_image_nodes])
        padded_nodes = [nodes + ([-1] * (max_num_nodes - len(nodes))) for nodes in per_image_nodes]

        padded_text_nodes = self.tokenizer.tokenize(
            merged_nodes + [PADDING_TAG]
        ) # TODO: Surely we dont need 77 tokens for a simple word

        return {
            'images': images,
            'captions': tokenized_captions,
            'captions_padding': padding_captions,
            'graph': {
                'nodes': tokenized_nodes,
                'text_nodes': padded_text_nodes,
                'edges': merged_edges,
                'node_types': tokenized_node_types,
                'edge_types': tokenized_edge_types,
                'batched_nodes': torch.tensor(
                    padded_nodes, dtype=torch.int64
                )
            }
        }