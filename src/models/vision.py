import torch.nn as nn
import torch
import open_clip

from src.data.data_defaults import CLIP_MODEL_TAG
from src.models.catr.utils import nested_tensor_from_tensor_list

class CLIPVisionEncoder(nn.Module):
    def __init__(self, clip_model_tag = CLIP_MODEL_TAG):
        super(CLIPVisionEncoder, self).__init__()

        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_tag, pretrained='laion2b_s34b_b79k')
        self.model = model.visual

    def forward(self, batch):
        return self.model(batch)

class CaTrBackbone(nn.Module):
    def __init__(self):
        super(CaTrBackbone, self).__init__()
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
        model.load_state_dict(
            torch.load(
                '/data/users/amolina/hmmkg/models/checkpoint_logged_last_pretrained.pth' # TODO: This should be an arg
            )['model']
        )
        self.backbone = model.backbone
        self.projection = model.input_proj
        self.encoder = model.transformer.encoder

    def forward(self, batch):

        images_list = [batch[i] for i in range(batch.shape[0])]
        nested = nested_tensor_from_tensor_list(images_list)
        # Credit to: saahiluppal/catr

        features, pos = self.backbone(nested)
        src, mask = features[-1].decompose()
        projected = self.projection(src)

        projected = projected.flatten(2).permute(2, 0, 1)
        pos_embed = pos[-1].flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        memory = self.encoder(projected, src_key_padding_mask=mask, pos=pos_embed)

        return memory