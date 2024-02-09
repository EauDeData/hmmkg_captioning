import open_clip
import math
from typing import *

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