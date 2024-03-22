import torch, torchvision
from PIL import Image
import networkx as nx

from src._io.args import parse_arguments
from src.data.data_defaults import IMAGENET_MEANS, IMAGENET_STDS, MYSELF_TAG
from src.data.datasets import CaptioningDataset
from src.data.collator import Collator

from main import prepare_models

'''
Example command: 
$python predict.py --image_encoder CaTr --image_emb_size 256 --decoder_depth 4 --decoder_width 4\
--decoder_architecture tr --gat_depth 1 --gat_width 2 --encoder_approach simple_tr_encoder --gat_feature_size 256

'''

args = parse_arguments()
model, graph_tokenizer, text_tokenizer = prepare_models(args)
model.load_state_dict(torch.load('it_wont_work_is_causal.pth'))
image_path = '/data/users/amolina/coco/val2017/000000000885.jpg'

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224), ),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)],)
image = Image.open(image_path)

the_void = nx.Graph()  # Trick the data collator into thinking this is a KG dataset
the_void.add_node(MYSELF_TAG, node_type='special', content=MYSELF_TAG)
the_void.add_node('None', node_type='event', content='None')
the_void.add_edge(MYSELF_TAG, 'None')

graph_data = CaptioningDataset.get_graph_data_from_path({'context': the_void},
                                                   [MYSELF_TAG, 'None'], args.nodes_to_text)

data = {'image': image, 'graph_data':graph_data,
         'caption': 'A photo of'}
batch = Collator(transforms, text_tokenizer, graph_tokenizer, use_sbert=False).simple_encoder_with_adj_collate(
    [data]
)
caption_template = batch['captions'][:, :2]


padding = torch.zeros_like(caption_template).float()
padding[:, -1] = float('-inf')

batch['captions'] = caption_template
batch['captions_padding'] = padding
print('-------------------input batch-------------')
print(batch)
print('-------------------output batch-------------')
print(model.eval_forward(batch))
