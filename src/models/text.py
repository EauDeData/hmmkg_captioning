import torch.nn as nn
import open_clip

from src.data.data_defaults import CLIP_MODEL_TAG
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

class TransformerTextDecoder(nn.Module):
    # Rebrà: Token de l'image encoder (imatge per fer captioning
        # Tokens de contexte extrets amb un context encoder
    pass
