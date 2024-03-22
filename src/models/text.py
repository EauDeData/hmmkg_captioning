import torch.nn as nn
import torch
from torch import Tensor
import open_clip
import math
import copy
from src.data.data_defaults import CLIP_MODEL_TAG
from src.models.attentions import ScaledDotProductAttention
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

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

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
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

class LSTMTextEncoder(nn.Module):
    def __init__(self, emb_size, n_layers):
        super(LSTMTextEncoder, self).__init__()

        self.lstm = nn.LSTM(emb_size, emb_size, num_layers=n_layers, batch_first=True)

    def forward(self, batch):

        output, (_, _) = self.lstm(batch)
        return output[:, -1, :]

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

        layer = nn.TransformerDecoderLayer(d_model=decoder_token_size, nhead=decoder_width, batch_first=False)
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_depth)
        self.lm_head = torch.nn.Linear(decoder_token_size, vocab_size)

        self.max_tokens_decode = max_tokens_during_train
        self.vocab_size = vocab_size
        self.d_size = decoder_token_size
        self.stop_token_id = stop_token
        self.start_token_id = start_token_id

        self.pos_emb = PositionalEncoding(decoder_token_size, dropout=0)

        self.text_processor = text_embedding.cpu()
        if train_text_emb:
            for param in self.text_processor.parameters(): param.requires_grad = False

        self.to(self.encoder.device)

    def eval_forward(self, X):
        encoder_dict_of_features = self.encoder(X)
        encoder_output = encoder_dict_of_features['features']  # Pass the batch X through the encoder
        memory = self.memory(encoder_output).permute(1,0,2)

        output = torch.ones(memory.shape[1], self.max_tokens_decode).long().to(memory.device) * self.start_token_id
        logits = torch.zeros(self.max_tokens_decode, memory.shape[1], self.vocab_size, device=memory.device)
        logits[0, :, self.start_token_id] = 1
        for t in range(1, self.max_tokens_decode):
            tgt_emb = self.pos_emb(self.text_processor(output[:, :t].detach()).transpose(1, 0))
            tgt_mask = generate_square_subsequent_mask(tgt_emb)


            decoder_output = self.gelu_fn(self.decoder(tgt=tgt_emb,
                                     memory=memory,
                                     tgt_mask=tgt_mask))

            pred_proba_t = self.lm_head(decoder_output)[-1, :, :]
            output_t = pred_proba_t.data.topk(1)[1].squeeze()
            output[:, t] = output_t
            logits[t] = pred_proba_t
        return {
            'features': memory,
            'language_head_output': logits,
            'generated_sequence': output,
            'hidden_states': None
        }

    def forward(self, X):

        encoder_dict_of_features = self.encoder(X)
        encoder_output = encoder_dict_of_features['features']  # Pass the batch X through the encoder
        memory = self.memory(encoder_output).permute(1,0,2)

        captions_templates = X['captions'].to(memory.device)

        sequence = self.text_processor(captions_templates)

        positional_sequence = self.pos_emb(sequence.permute(1,0,2))
        mask = generate_square_subsequent_mask(positional_sequence)

        decoded = self.gelu_fn(self.decoder(
            tgt=positional_sequence,
            memory=memory,
            tgt_key_padding_mask=(X['captions_padding'] == float('-inf')).to(memory.device),
            tgt_mask=mask,
        ))

        # Project the decoder output to vocabulary space
        output = self.lm_head(decoded)

        return {
            'features': memory,
            'language_head_output': output,
            'generated_sequence': output.argmax(-1).transpose(1, 0),
            'hidden_states': None
        }

    def autorrecurrent_forward(self, batch):
        raise NotImplementedError('For training autorecurring, implement it. For eval, use eval_forward()')


class GPT2Decoder(nn.Module):
    def __init__(self, encoder, encoder_input_size, decoder_token_size, decoder_depth, vocab_size,
                 decoder_width, text_embedding,
                 max_tokens_during_train=100, stop_token=0, start_token_id=0, train_text_emb = True):
        super(GPT2Decoder, self).__init__()

        self.encoder = encoder
        self.memory = torch.nn.Linear(encoder_input_size, decoder_token_size)
        self.gelu_fn = torch.nn.GELU()

        self.max_tokens = max_tokens_during_train

        configuration = GPT2Config(vocab_size=vocab_size, n_positions=max_tokens_during_train,
                                   n_embd=decoder_token_size, n_layer=decoder_depth, n_head=decoder_width,
                                   bos_token_id=start_token_id, eos_token_id=stop_token, pad_token_id=0)
        self.gpt2 = GPT2LMHeadModel(configuration)
        self.start_token_id = start_token_id
        self.embedding = text_embedding.cpu()
        self.pos_emb = PositionalEncoding(decoder_token_size, dropout=0)
        self.gpt2.config.is_decoder = True  # Set as decoder
        self.start_embedding = torch.nn.Embedding(1, decoder_token_size)

        if train_text_emb: # Actually it is if freeze, im stupid
            for param in self.embedding.parameters():
                param.requires_grad = False

        self.to(self.encoder.device)

    def forward(self, X):

        encoder_dict_of_features = self.encoder(X)
        encoder_output = encoder_dict_of_features['features']  # Pass the batch X through the encoder
        memory = self.memory(encoder_output).transpose(1,0) # (SEQ, BATCH, DIM)
        end_of_memory_token = self.start_embedding(
            torch.zeros((1, memory.shape[1]), device = memory.device, dtype=torch.int64)
        )
        memory_with_start_token = torch.cat(
            (memory, end_of_memory_token), dim=0
        )

        positional_memory = self.pos_emb(memory_with_start_token).transpose(1,0)
        decoder_output = self.gpt2.generate(input_ids=None, attention_mask=None,
                                               past_key_values=positional_memory)
        print(decoder_output)
        exit()
        logits = decoder_output.logits








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
                 start_token_id=0, max_tokens_during_train=100, decoder_heads=1, decoder_depth=1):
        super(LSTMDecoderWithAttention, self).__init__()

        self.encoder = encoder
        self.decoder_token_size = decoder_token_size

        self.max_tokens = max_tokens_during_train

        self.features_attn_head = nn.Sequential(nn.ReLU(), nn.Linear(encoder_input_size, decoder_token_size))

        self.lm_head = nn.Linear(decoder_token_size, vocab_size)

        self.vocab_size = vocab_size
        self.start_token_id = start_token_id

        self.lstm = torch.nn.LSTM(decoder_token_size, decoder_token_size, batch_first=False)

        encoder_layers = torch.nn.TransformerEncoderLayer(decoder_token_size, decoder_heads, decoder_token_size,
                                                          0.1, batch_first=True)
        self.transformer_encoder_context = torch.nn.TransformerEncoder(encoder_layers, decoder_depth)
        self.pos_emb = PositionalEncoding(decoder_token_size, 0)

        self.initial_states = torch.nn.Embedding(2, decoder_token_size)

        self.to(encoder.device)

    def eval_forward(self, batch):
        return self.forward(batch)
    def forward(self, batch):

        features = self.encoder(batch)['features'] # (BS, SEQ_SIZE, ...)
        projected_features = self.features_attn_head(features)

        contextualized_features = self.transformer_encoder_context(projected_features)\
                                      .transpose(1,0)[:self.max_tokens] #(SEQ_SIZE, BS, ...)
        h0, c0 = self.initial_states(torch.zeros(features.shape[0], dtype=torch.int, device=features.device)),\
            self.initial_states(torch.ones(features.shape[0], dtype=torch.int, device=features.device))

        positional_encoded_context = self.pos_emb(contextualized_features)

        recurrent_output, hidden_states = self.lstm(positional_encoded_context, (h0[None, :], c0[None, :]))

        output = self.lm_head(recurrent_output)

        return {
            'features': features,
            'language_head_output': output,
            'generated_sequence': output.argmax(-1).transpose(1, 0),
            'hidden_states': hidden_states
        }

    def eval_forward(self, *args, **kwargs):
        return self(*args, **kwargs)
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
