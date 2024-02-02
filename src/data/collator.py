import torch
class Collator:
    '''

    A set of collator functinos so we can select them according to the needed approach

    '''

    def __init__(self, transforms, text_tokenizer):
        self.transforms = transforms
        self.tokenizer = text_tokenizer
        return

    def base_collate_captioning(self, batch):
        images = torch.stack([self.transforms(sample['image']) for sample in batch])
        tokenized_captions = torch.stack([self.tokenizer.tokenize(sample['caption']) for sample in batch]).view\
            (images.shape[0], -1) # (BATCH_SIZE, context_length)
        print(tokenized_captions.shape, images.shape)

        # NOW we need to use the graph_element tokenizer from node_and_edge tokenizers
        # There we will get the IDs (numbers) from nodes, node categories and edge categories
