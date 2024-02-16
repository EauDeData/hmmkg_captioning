from src.data.datasets import CaptioningDataset
from src.models.vision import CaTrBackbone

import resource
import torch
import numpy as np
from PIL import Image

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print('rlimit', rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

dataset = CaptioningDataset(3, 3, split = 'train')

model = CaTrBackbone()
print(model(torch.rand(8, 3, 224, 224)).shape)