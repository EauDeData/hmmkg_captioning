VALID_CATEGORIES = ['person', 'institution', 'location', 'participation', 'role', 'event_side', 'date', 'event']
# And we will need to add images as processed by text processor
MYSELF_TAG = '[MYSELF_NODE]'
PADDING_TAG = '[GRAPH_PADDING]'
TEXT_PROCESSOR_NODE_CATEGORIES = [] # TODO: In the future
# "text content will be here as well, no?
EMBEDDING_NODE_CATEGORIES = VALID_CATEGORIES + ['special']

PATH_TO_GRAPH_GEXF = '/data/users/amolina/hmmkgv2/final_graph_v1.gexf'
PATH_TO_IMAGES_TSV = '/data/users/amolina/hmmkgv2/images/images_split.tsv'
IMAGES_PARENT_FOLDER = '/data/users/amolina/hmmkgv2/images/images/'
DATASET_SAVE_PATH_FSTRING = '/data/users/amolina/hmmkgv2/{}_dataset_checkpoint_V2.0.pkl'
NON_ADMITED_FILE_FORMATS = ['djvu', 'flac', 'gz', 'mp3', 'oga', 'ogg', 'ogv', 'webm']

CLIP_MODEL_TAG = 'ViT-B-32'
DEFAULT_TEXT_TOKENIZER_CONTEXT_LENGTH = 77

GRAPH_TOKENIZER_DEFAULT_PATH = '/data/users/amolina/hmmkgv2/graph_tokenizer.json'

IMAGENET_MEANS = [0.485, 0.456, 0.406]  # RGB mean values
IMAGENET_STDS = [0.229, 0.224, 0.225]   # RGB standard deviation values

VOCAB_WEIGHTS = './output_viz/vocab_weights.json'