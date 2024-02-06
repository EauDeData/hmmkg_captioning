import torch.nn as nn
import open_clip

from src.data.data_defaults import CLIP_MODEL_TAG
class CLIPVisionEncoder(nn.Module):
    def __init__(self, clip_model_tag = CLIP_MODEL_TAG):
        super(CLIPVisionEncoder, self).__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_tag, pretrained='laion2b_s34b_b79k')
        self.model = model.visual

    def forward(self, batch):
        return self.model(batch)