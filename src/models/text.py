import torch.nn as nn
class CLIPTextEncoder:
    pass

class TransformerTextEncoder(nn.Module):
    def __init__(self, text_embedding):
        # Per ser consistents, hem de tokenitzar fora i fer els embeddings al forward
        pass

class TransformerTextDecoder(nn.Module):
    # Rebrà: Token de l'image encoder (imatge per fer captioning
        # Tokens de contexte extrets amb un context encoder
    pass
