from src.data.datasets import CaptioningDataset
from src.data.collator import Collator
from src.tokenizers.tokenizers import CLIPOriginalTokenizer
from src.tokenizers.node_and_edges_tokenizers import GraphTokenizer
from src.data.data_defaults import PATH_TO_GRAPH_GEXF, IMAGENET_MEANS, IMAGENET_STDS
from src.models.graphs import GraphContextGAT
from src.models.text import CLIPTextEncoder, TransformerDecoder
from src.models.vision import CLIPVisionEncoder
from src.loops.cross_entropy_train import cross_entropy_train_loop
from src.loops.eval import eval

from torch.utils.data import DataLoader
import torchvision
import torch
import networkx as nx
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
print('rlimit', rlimit)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

graph_tokenizer = GraphTokenizer(PATH_TO_GRAPH_GEXF)

tokenizer = CLIPOriginalTokenizer(context_length = 77) # While using the CLIP encoder, the context must be 77

dataset = CaptioningDataset(3, 3, split = 'train')
transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224),),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)],)

collator = Collator(transforms, tokenizer, graph_tokenizer)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collator.base_collate_captioning,
                        num_workers=12)

feature_size = 256
text_encoder = CLIPTextEncoder()
print(text_encoder.model.token_embedding)

graph_model = GraphContextGAT(CLIPVisionEncoder(), 512, text_encoder, 512, feature_size,
                        torch.nn.Embedding(len(graph_tokenizer.token_dict), feature_size // 2),
                        1, 8, 256, 128, device='cuda')
graph_model.train()
model = TransformerDecoder(graph_model, feature_size, text_encoder.model.token_embedding, 512, 1, len(tokenizer),
                           1, max_tokens_during_train=77, stop_token=tokenizer.eos_token_id)
print(tokenizer.bos_token_id)

eval(dataloader, model, tokenizer)