VALID_CATEGORIES = ['person', 'institution', 'location', 'participation', 'role', 'event_side', 'date', 'event']
# And we will need to add images as processed by text processor
MYSELF_TAG = '[MYSELF_NODE]'
PADDING_TAG = '[GRAPH_PADDING]'
TEXT_PROCESSOR_NODE_CATEGORIES = ['participation', 'role', 'event', 'date'] # TODO: In the future
# "text content will be here as well, no?
EMBEDDING_NODE_CATEGORIES = ['person', 'institution', 'location', 'event_side'] + ['special']

PATH_TO_GRAPH_GEXF = '/data/users/amolina/hmmkg/knowledge_uncurated_merged_graph.gexf'
PATH_TO_IMAGES_TSV = '/data/users/amolina/hmmkg/images_split.tsv'
IMAGES_PARENT_FOLDER = '/data/users/amolina/hmmkg/images/'
DATASET_SAVE_PATH_FSTRING = '/data/users/amolina/hmmkg/{}_dataset_checkpoint.pkl'
NON_ADMITED_FILE_FORMATS = ['djvu', 'flac', 'gz', 'mp3', 'oga', 'ogg', 'ogv', 'webm']

CLIP_MODEL_TAG = 'ViT-B-32'
DEFAULT_TEXT_TOKENIZER_CONTEXT_LENGTH = 100

GRAPH_TOKENIZER_DEFAULT_PATH = '/data/users/amolina/hmmkg/graph_tokenizer.json'

IMAGENET_MEANS = [0.485, 0.456, 0.406]  # RGB mean values
IMAGENET_STDS = [0.229, 0.224, 0.225]   # RGB standard deviation values