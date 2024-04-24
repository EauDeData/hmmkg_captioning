import open_clip
import math
from typing import *
from transformers import AutoTokenizer
import torch

from src.data.data_defaults import CLIP_MODEL_TAG, DEFAULT_TEXT_TOKENIZER_CONTEXT_LENGTH
class CLIPOriginalTokenizer:
    '''
        The text we need to encode for now is not any complicated,
        just download the CLIP tokenizer so we can get our thing done.
        Further tokenizers will follow the same i/o regime.
    '''
    def __init__(self, clip_model_tag = CLIP_MODEL_TAG,
                 context_length = DEFAULT_TEXT_TOKENIZER_CONTEXT_LENGTH,*args):
        self.tokenizer = open_clip.get_tokenizer(clip_model_tag)

        self.bos_token_id = self.tokenizer.sot_token_id
        self.eos_token_id = self.tokenizer.eot_token_id

        self.sentence_len = context_length

    def __len__(self):
        return self.tokenizer.vocab_size

    def tokenize(self, sentences: list):
        return self.tokenizer(sentences, context_length=self.sentence_len) # TODO: Context length should be a parameter
    def decode(self, single_sentence: List[int]):
        return self.tokenizer.decode(single_sentence)


class BERTTokenizer:
    def __init__(self, context_length = DEFAULT_TEXT_TOKENIZER_CONTEXT_LENGTH):
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.special_tokens = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]
        self.tokenizer.add_tokens([f"[{x}]" for x in self.special_tokens])
        assert len(self.tokenizer.tokenize('[WORK_OF_ART]')) == 1, "look at this happening to the tokenizer!"

        self.sentence_len = context_length

        self.bos_token_id = 101
        self.eos_token_id = 102

    def __len__(self):
        return self.tokenizer.__len__()

    def tokenize(self, sentences: list):
        return torch.stack([torch.tensor(self.tokenizer.encode(sent, max_length=self.sentence_len, truncation=True,
                                                    padding='max_length',))
                           for sent in sentences])


    def decode(self, single_sentence: List[int]):
        return self.tokenizer.decode(single_sentence)
