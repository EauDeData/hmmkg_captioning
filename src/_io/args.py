from src.data.data_defaults import (VALID_CATEGORIES,
                                    TEXT_PROCESSOR_NODE_CATEGORIES,
                                    EMBEDDING_NODE_CATEGORIES,
                                    PATH_TO_GRAPH_GEXF,
                                    PATH_TO_IMAGES_TSV,
                                    IMAGES_PARENT_FOLDER,
                                    DATASET_SAVE_PATH_FSTRING,
                                    NON_ADMITED_FILE_FORMATS,
                                    CLIP_MODEL_TAG,
                                    DEFAULT_TEXT_TOKENIZER_CONTEXT_LENGTH,
                                    GRAPH_TOKENIZER_DEFAULT_PATH)

import argparse
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

def assert_preconditions(args):
    if args.text_encoder == 'CLIP':
        assert args.text_context_size == 77, "When using CLIP, context must be set to 77"
        assert args.text_emb_size==512, "CLIP uses 512 embedding size (text)"
        assert args.decoder_emb_size==512, 'CLIP projects text to 512 features, so does decoder'

    if args.image_encoder == 'CLIP':
        assert args.image_emb_size == 512, "CLIP uses 512 embedding size (image)"
    elif args.image_encoder == 'CaTr':
        assert args.image_emb_size == 256

    assert not (args.freeze_backbone and (args.train_vision and args.train_text)), 'what the fuck'
    assert not (not args.freeze_backbone and (args.train_vision and args.train_text)), 'what the fuck'


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--log_wandb', action='store_true')
    parser.add_argument('--use_sbert', action='store_true')
    parser.add_argument('--optimizer', default='RMSprop', type=str,
                        choices=['Adam', 'AdamW', 'RMSprop'])
    parser.add_argument('--dataset', choices=['hmmkg', 'coco', 'gcc'], default='hmmkg')

    parser.add_argument('--coco_base_dir', default='/data/users/amolina/coco/')
    parser.add_argument('--gcc_base_dir',
                        default='/data/users/amolina/ccaptions/DownloadConceptualCaptions')

    parser.add_argument('--random_walk_len', type=int, default=12)
    parser.add_argument('--context_neight_depth', type=int, default=4)

    parser.add_argument('--dataset_save_path_fstring', type=str, default=DATASET_SAVE_PATH_FSTRING)
    parser.add_argument('--path_to_gexf_graph', type=str, default=PATH_TO_GRAPH_GEXF)
    parser.add_argument('--path_to_images_tsv', type=str, default=PATH_TO_IMAGES_TSV)
    parser.add_argument('--images_parent_folder', type=str, default=IMAGES_PARENT_FOLDER)
    parser.add_argument('--graph_tokenizer_path', type=str, default=GRAPH_TOKENIZER_DEFAULT_PATH)
    parser.add_argument('--text_tokenizer', choices=['CLIP', 'bert'], default='bert')

    parser.add_argument('--non_admited_file_formars', nargs='+', type=str,
                        default=NON_ADMITED_FILE_FORMATS)

    parser.add_argument('--valid_categories', nargs='+', type=str, default=VALID_CATEGORIES)
    parser.add_argument('--nodes_to_text', nargs='+', type=str, default=TEXT_PROCESSOR_NODE_CATEGORIES)
    parser.add_argument('--nodes_to_embedding', nargs='+', type=str, default=EMBEDDING_NODE_CATEGORIES)

    parser.add_argument('--image_encoder', type=str, default='CLIP', choices=['CLIP', 'CaTr'])
    parser.add_argument('--text_encoder', type=str, default='CLIP', choices=['CLIP', 'projection', 'lstm'])
    parser.add_argument('--text_context_size', type=int, default=DEFAULT_TEXT_TOKENIZER_CONTEXT_LENGTH)

    parser.add_argument('--clip_tag', type=str, default=CLIP_MODEL_TAG)
    parser.add_argument('--image_emb_size', type=int, default=512)
    parser.add_argument('--text_emb_size', type=int, default=512)

    parser.add_argument('--encoder_approach', type=str, default='simple_tr_encoder',
                        choices=['simple_tr_encoder', 'gat_encoder']
                        )
    parser.add_argument('--gat_feature_size', type=int, default=256)
    parser.add_argument('--gat_depth', type=int, default=1)
    parser.add_argument('--gat_width', type=int, default=8)
    parser.add_argument('--gat_in_channels', type=int, default=256)
    parser.add_argument('--gat_hidden_channels', type=int, default=128)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--train_vision', action='store_true')
    parser.add_argument('--train_text', action='store_true')


    parser.add_argument('--decoder_architecture', type=str, default='tr', choices=['tr', 'lstm', 'gpt2'])
    parser.add_argument('--decoder_emb_size', type=int, default=512)
    parser.add_argument('--decoder_depth', type=int, default=1)
    parser.add_argument('--decoder_width', type=int, default=8)
    parser.add_argument('--auto_recurrent_decoder', action='store_true')

    parser.add_argument('--load_global_checkpoint', type=str, default=None)
    parser.add_argument('--save_checkpoint_to', type=str, default=None)

    parser.add_argument('--use_cross_entropy_weights', action='store_false')

    arguments = parser.parse_args()
    assert_preconditions(arguments)
    return arguments
