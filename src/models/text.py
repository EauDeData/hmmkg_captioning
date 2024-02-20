import torch.nn as nn
import torch
from torch import Tensor
import open_clip
import math
import copy
from src.data.data_defaults import CLIP_MODEL_TAG
from src.models.attentions import ScaledDotProductAttention
import torch.nn.functional as F

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    sz_shape = sz.shape[0] # Sequence size
    mask = (torch.triu(torch.ones(sz_shape, sz_shape)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(sz.device)

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
                 max_tokens_during_train=100, stop_token = 0, start_token_id=0, train_text_emb=False,
                 auto_recurrent=False):
        super(TransformerDecoder, self).__init__()

        self.encoder = encoder
        self.memory = torch.nn.Linear(encoder_input_size, decoder_token_size)

        self.gelu_fn = torch.nn.GELU()

        self.layer = nn.TransformerDecoderLayer(d_model=decoder_token_size, nhead=decoder_width)
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=decoder_depth)

        self.lm_head = torch.nn.Linear(decoder_token_size, vocab_size)

        self.cls_token = torch.zeros(1, decoder_token_size, device=encoder.device)

        self.max_tokens_decode = max_tokens_during_train
        self.vocab_size = vocab_size
        self.d_size = decoder_token_size
        self.stop_token_id = stop_token
        self.start_token_id = start_token_id

        self.pos_emb = PositionalEncoding(decoder_token_size, dropout=0, max_len=max_tokens_during_train + 1)
        #self.pos_emb = torch.nn.Linear(1, max_tokens_during_train * self.d_size) # Acts as a parameter\
        # for batching porpouses

        self.embedding = text_embedding.cpu()

        if train_text_emb: # Actually it is if freeze, im stupid
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.to(self.encoder.device)

        self.forward = self.autorrecurrent_forward if auto_recurrent else self.non_autorecurrent_forward

    def eval_forward(self, X):

        encoder_dict_of_features = self.encoder(X)
        encoder_output = encoder_dict_of_features['features']  # Pass the batch X through the encoder
        memory = self.memory(encoder_output).transpose(1, 0)

        # Initialize the input for autoregressive decoding
        cls_token = torch.tensor([[0]] * memory.size(1), dtype=torch.long, device=memory.device).\
            transpose(1, 0) # Do start of sequence maybe
        input_sequence = torch.zeros(self.max_tokens_decode+1, memory.size(1), dtype=torch.int64, device=memory.device)
        input_sequence[0] = cls_token
        mask = generate_square_subsequent_mask(input_sequence)

        output_logits = torch.empty(memory.size(1), self.max_tokens_decode, self.vocab_size, device=memory.device)
        # output_logits[:, 0, self.start_token_id] = 1
        # Autoregressive generation loop

        for idx in range(self.max_tokens_decode):

            sequence = self.embedding(input_sequence.detach())
            positional_sequence = self.pos_emb(sequence)
            key_padding_mask = torch.zeros_like(input_sequence, dtype=torch.float32, device=memory.device)
            key_padding_mask[(idx+1):] = float('-inf')

            decoded = self.gelu_fn(self.decoder(
                tgt=positional_sequence, # SEQ_SIZE; BSIZE; EMB_SIZE
                memory=memory, # M_SEQ_SIZE; B_SIZE; EMB_SIZE
                tgt_mask=mask, # B_SIZE; SEQ_SIZE, SEQ_SIZE,
                key_padding_mask=key_padding_mask
            ))

            # Project the decoder output to vocabulary space
            output = self.lm_head(decoded[idx + 1])  # Take the last time step's output
            output_logits[:, idx] = output

            # Get the predicted token (argmax)
            predicted_token = output.argmax(dim=-1)

            # Concatenate the predicted token to the input sequence for the next iteration
            input_sequence[idx + 1] = predicted_token

        # Return the generated sequence (excluding the start token)
        generated_sequence = input_sequence[1:]

        return {
            'features': memory,
            'generated_sequence': generated_sequence,
            'language_head_output': output_logits,
            'hidden_states': None
        }

    def non_autorecurrent_forward(self, X):

        encoder_dict_of_features = self.encoder(X)
        encoder_output = encoder_dict_of_features['features']  # Pass the batch X through the encoder
        memory = self.memory(encoder_output).transpose(1,0)

        sequence = self.embedding(X['captions'].to(memory.device))

        cls = torch.zeros(sequence.shape[0], sequence.shape[-1], device=sequence.device)[None, :, :]
        positional_sequence = self.pos_emb(torch.cat((cls, sequence.transpose(1, 0)), dim=0))
        mask = generate_square_subsequent_mask(positional_sequence)

        decoded = self.gelu_fn(self.decoder(
            tgt=positional_sequence,
            memory=memory,
            tgt_key_padding_mask=X['captions_padding'].to(memory.device),
            tgt_mask=mask

        ))

        # Project the decoder output to vocabulary space
        output = self.lm_head(decoded)[1:]

        return {
            'features': memory,
            'language_head_output': output,
            'generated_sequence': output.argmax(-1).transpose(1, 0),
            'hidden_states': None
        }

    def autorrecurrent_forward(self, batch):
        raise NotImplementedError('For training autorecurring, implement it. For eval, use eval_forward()')


def scaled_dot_product_attention(query, key, value):
    # Compute dot product between query and key

    dot_product = torch.matmul(query, key.transpose(2, 1))

    # Scale the dot product
    scaled_dot_product = dot_product / torch.sqrt(torch.tensor(query.size(-1), dtype=torch.float32))

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scaled_dot_product, dim=-1)

    # Multiply attention weights by values
    attention_output = torch.matmul(attention_weights.transpose(2, 1), value)

    return attention_output, attention_weights

class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, encoder, encoder_input_size, text_embedding, decoder_token_size, vocab_size,
                 start_token_id=0, max_tokens_during_train=100):
        super(LSTMDecoderWithAttention, self).__init__()

        self.encoder = encoder
        self.decoder_token_size = decoder_token_size

        self.bos_token = text_embedding(torch.tensor([start_token_id], device=encoder.device))
        self.max_tokens = max_tokens_during_train

        self.features_attn_head = nn.Sequential(nn.ReLU(), nn.Linear(encoder_input_size, decoder_token_size))
        self.text_features_attn_head = nn.Sequential(nn.ReLU(), nn.Linear(decoder_token_size, decoder_token_size))
        self.features_values = nn.Sequential(nn.ReLU(), nn.Linear(encoder_input_size, decoder_token_size))

        self.lm_head = nn.Linear(decoder_token_size, vocab_size)

        self.vocab_size = vocab_size
        self.start_token_id = start_token_id

        self.lstm_cell = torch.nn.LSTMCell(decoder_token_size, decoder_token_size)
        self.attention_layer = scaled_dot_product_attention


        self.to(encoder.device)

    def eval_forward(self, batch):
        return self.forward(batch)
    def forward(self, batch):

        features = self.encoder(batch)['features']

        context_key = self.features_attn_head(features)
        context_values = self.features_values(features)

        output_seq = torch.zeros((self.max_tokens, features.shape[0], self.vocab_size),
                                device=self.encoder.device)

        output_seq[0, :, self.start_token_id] = 1

        hidden_state = torch.stack([self.bos_token.squeeze().clone().detach()] * features.shape[0])

        sequence_ctx = torch.zeros(features.shape[0], self.decoder_token_size, device=self.encoder.device)


        for seq_idx in range(1, self.max_tokens):

            input_values = hidden_state.detach().clone()
            input_seq_keys = self.text_features_attn_head(input_values)[:, None, :]

            context, attn = self.attention_layer(context_key, input_seq_keys, context_values) # This gives us
            # context depending on the sequence time

            #print(context.shape, hidden_state.shape)
            hidden_state, sequence_ctx = self.lstm_cell(context[:, 0], (hidden_state, sequence_ctx))
            output_seq[seq_idx] = self.lm_head(hidden_state)

        return {'language_head_output': output_seq}


class CaTrDecoder(nn.Module):
    def __init__(self, encoder):
        super(CaTrDecoder).__init__()
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

        self.encoder = encoder
        self.embeddings = model.transformer.embeddings
        self.decoder = model.transformer.decoder

    def forward(self, batch):

        encoder_dict_of_features = self.encoder(batch)
        encoder_output = encoder_dict_of_features['features']  # Pass the batch X through the encoder
        memory = self.memory(encoder_output).transpose(1,0)
