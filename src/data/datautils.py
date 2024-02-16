from src.data.data_defaults import VOCAB_WEIGHTS

import torch
import json
import os
from tqdm import tqdm

def compute_or_get_vocab_weights(dataset, tokenizer, padding_token_id, vocab_size, path=VOCAB_WEIGHTS):


    if os.path.exists(path):
        token_dict = json.load(open(path, 'r'))

    else:
        token_dict = {}

        for data in tqdm(dataset.data_items, desc='Computing loss weights from vocab...'):

            tokenized_data = tokenizer.tokenize(list(data['captions'])).cpu().numpy().tolist()

            for token in sum(tokenized_data, start=[]):
                if not token in token_dict: token_dict[token] = 0
                token_dict[token] += 1
                if token == tokenizer.eos_token_id: break

        total_tokens = sum(token_dict.values())
        token_dict = {key: value / total_tokens for key, value in token_dict.items()}
        json.dump(token_dict, open(path, 'w'))

    weights = torch.zeros(vocab_size)
    for key, weight in token_dict.items():
        weights[int(key)] = 1 - weight
    weights[padding_token_id] = 0
    weights[tokenizer.bos_token_id] = 1
    weights[tokenizer.eos_token_id] = 1
    return weights




