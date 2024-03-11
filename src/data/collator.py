import torch
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from src.data.data_defaults import MYSELF_TAG
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

        # adj_matrix = torch.stack([torch.from_numpy(sample['graph_data']['adj']) for sample in batch])

        tokenized_captions = torch.stack([self.tokenizer.tokenize(sample['caption']) for sample in batch]).view\
            (images.shape[0], -1) # (BATCH_SIZE, context_length)

        ## NODE EMBEDDINGS COLLATED
        node_embs_idxs = [[sample['graph_data']['to_node_emb'][idx]['global_idx']
                            for idx in sample['graph_data']['to_node_emb']]
                           for sample in batch]

        node_embs_tokenized = [[self.graph_tokenizer.token_dict[idx]
                            for idx in sample['graph_data']['to_node_emb']]
                           for sample in batch]

        node_embs_categories_tokenized = [[self.graph_tokenizer.token_dict[
                                                sample['graph_data']['to_node_emb'][idx]['node_type']
                                            ]
                            for idx in sample['graph_data']['to_node_emb']]
                           for sample in batch]

        ## NODE-TEXT EMBEDDINGS COLLATED
        node_txt_idxs = [[sample['graph_data']['to_text_emb'][idx]['global_idx']
                           for idx in sample['graph_data']['to_text_emb']]
                          for sample in batch]

        node_txt_tokenized = [[self.tokenizer.tokenize(
                                                sample['graph_data']['to_text_emb'][idx]['content']
                            )
                            for idx in sample['graph_data']['to_text_emb']]
                           for sample in batch] # WARNING: Padding depends on the tokenizer, so text is already padded

        node_txt_categories_tokenized = [[self.graph_tokenizer.token_dict[
                                                sample['graph_data']['to_text_emb'][idx]['node_type']
                                            ]
                            for idx in sample['graph_data']['to_text_emb']]
                           for sample in batch]

        ## EDGES COLLATED

        edges_idxs_coo = [[(idx['global_index_src'], idx['global_index_dst'])
                           for idx in sample['graph_data']['edges']]
                          for sample in batch]

        edges_categories = [[self.graph_tokenizer.token_dict[idx['edge_type']]
                           for idx in sample['graph_data']['edges']]
                          for sample in batch]

        return {
            'images': images,
            'captions': tokenized_captions,
            'graph': {
                'node_embs': {
                    'idxs': node_embs_idxs,
                    'tokens': node_embs_tokenized,
                    'categories': node_embs_categories_tokenized
                },
                'node_txts': {
                    'idxs': node_txt_idxs,
                    'tokens': node_txt_tokenized,
                    'categories': node_txt_categories_tokenized
                },
                'edges': {
                    'idxs_coo': edges_idxs_coo,
                    'categories': edges_categories
                }
            }
        }