import torch.nn as nn
import torch
from torch import Tensor
import open_clip
import math
import copy
from src.data.data_defaults import CLIP_MODEL_TAG

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model_tag = CLIP_MODEL_TAG):
        super(CLIPTextEncoder, self).__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_tag, pretrained='laion2b_s34b_b79k')
        self.model = model

    def forward(self, batch):
        return self.model.encode_text(batch)

class TransformerTextEncoder(nn.Module):
    def __init__(self, text_embedding):
        # Per ser consistents, hem de tokenitzar fora i fer els embeddings al forward
        pass

class TransformerDecoder(nn.Module):
    def __init__(self, encoder, encoder_input_size, text_embedding, decoder_token_size, decoder_depth, vocab_size,
                 decoder_width,
                 max_tokens_during_train=100, stop_token = 0, train_text_emb=False):
        super(TransformerDecoder, self).__init__()

        self.encoder = encoder
        self.memory = torch.nn.Linear(encoder_input_size, decoder_token_size)

        self.gelu_fn = torch.nn.GELU()

        self.layer = nn.TransformerDecoderLayer(d_model=decoder_token_size, nhead=decoder_width)
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=decoder_depth)

        self.lm_head = torch.nn.Linear(decoder_token_size, vocab_size)

        self.cls_token = torch.nn.Parameter(torch.rand(1, decoder_token_size))

        self.pos_emb = PositionalEncoding(decoder_token_size, dropout=0, max_len=max_tokens_during_train)
        self.max_tokens_decode = max_tokens_during_train
        self.vocab_size = vocab_size
        self.d_size = decoder_token_size
        self.stop_token_id = stop_token

        self.embedding = copy.deepcopy(text_embedding.cpu())

        if not train_text_emb:
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.to(self.encoder.device)

    def forward(self, X):

        encoder_output = self.encoder(X)['features']  # Pass the batch X through the encoder
        memory = self.memory(encoder_output).transpose(1,0)
        sequence = torch.empty((self.max_tokens_decode+1, memory.shape[1], self.d_size), device=self.encoder.device)
        sequence[0, :, :] = self.cls_token
        for sequence_id in range(1, self.max_tokens_decode + 1):

            target_sequence = self.pos_emb(sequence[:sequence_id, :, :])

            decoded = self.gelu_fn(self.decoder(
                tgt=target_sequence,
                memory=memory))

            lang_head_output = self.lm_head(decoded)[-1, :, :]

            most_likely_tokens = self.embedding(torch.argmax(lang_head_output, 1)).detach() # TODO: Observar\
            # si això és un perill
            sequence[sequence_id, :, :] = most_likely_tokens.unsqueeze(0)

        return {
            'features': memory,
            'language_head_output': self.lm_head(sequence[1:]), # El token 0 és el CLS
            'hidden_states': sequence[1:]
        }
