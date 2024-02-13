import torch
from torch.nn.utils.rnn import pad_sequence
class Collator:
    '''

    A set of collator functinos so we can select them according to the needed approach

    '''

    def __init__(self, transforms, text_tokenizer, graph_tokenizer, padding_token = 0):
        self.transforms = transforms
        self.tokenizer = text_tokenizer
        self.graph_tokenizer = graph_tokenizer
        self.padding_token = 0
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

        print([[sample['graph_data'][node_type][idx]['global_idx']
                        for idx in sample['graph_data'][node_type]]
                       for sample in batch for node_type in ['to_node_emb', 'to_text_emb']])
        exit()
        all_adjs = torch.zeros(len(batch), max_nodes, max_nodes)
        all_nodes = sorted([[sample['graph_data']['to_node_emb'][idx]
                        for idx in sample['graph_data'][node_type]]
                       for sample in batch for node_type in ['to_node_emb', 'to_text_emb']],
               key=lambda x: x['global_idx'])
        print(all_nodes)




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
        print([[idx
                            for idx in sample['graph_data']['to_node_emb']]
                           for sample in batch])
        exit()
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