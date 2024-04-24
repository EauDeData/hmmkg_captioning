import numpy as np

from src.data.datasets import MaskedCaptionningDataset, CocoCaption, GCCaptions
from src._io.ioutils import read_json
from src.data.collator import Collator
from src.tokenizers.tokenizers import CLIPOriginalTokenizer, BERTTokenizer
from src.tokenizers.node_and_edges_tokenizers import GraphTokenizer
from src.data.data_defaults import IMAGENET_MEANS, IMAGENET_STDS
from src.data.datautils import compute_or_get_vocab_weights
from src.models.graphs import GraphContextGAT, GraphContextTransformerEncoder
from src.models.text import (CLIPTextEncoder, TransformerDecoder,
                             LSTMDecoderWithAttention, GPT2Decoder, LSTMTextEncoder, TransformerTextEncoder,
                             TwoStagesTransformerDecoder)
from src.models.vision import CLIPVisionEncoder, CaTrBackbone
from src.loops.cross_entropy_train import cross_entropy_train_loop, cross_entropy_train_loop_for_masked_filling
from src.loops.eval import eval

from torch.utils.data import DataLoader


import torch
import torchvision
import os

torch.manual_seed(42)

def get_graph_embedding(args):
    graph_tokenizer, text_tokenizer = get_tokenizers(args)
    return graph_tokenizer, text_tokenizer, torch.nn.Embedding(len(graph_tokenizer.token_dict),
                                                               args.gat_feature_size),\
        torch.nn.Embedding(len(text_tokenizer), args.text_emb_size)

def get_image_encoder(args):
    if args.image_encoder == 'CLIP':
        return CLIPVisionEncoder(clip_model_tag=args.clip_tag)
    elif args.image_encoder == 'CaTr':
        return CaTrBackbone()
    else:
        raise NotImplementedError(f"{args.image_encoder} is not among implemented image encoders")

def get_text_encoder(args, tokenizer = None):
    if args.text_encoder == 'CLIP':
        return CLIPTextEncoder(clip_model_tag=args.clip_tag)
    elif args.text_encoder == 'projection':
        return torch.nn.Sequential(torch.nn.Linear(args.text_emb_size, args.text_emb_size), torch.nn.ReLU())
    elif args.text_encoder == 'lstm':
        return torch.nn.Sequential(LSTMTextEncoder(args.text_emb_size, args.gat_depth), torch.nn.ReLU())
    elif args.text_encoder == 'tr':
        return TransformerTextEncoder(len(tokenizer), args.text_emb_size, args.gat_width, args.gat_depth )

    else:
        raise NotImplementedError(f"{args.text_encoder} is not among implemented image encoders")
def prepare_models(args):

    graph_tokenizer, text_tokenizer, graph_embedding, text_embedding = get_graph_embedding(args)
    visual_model, textual_model = get_image_encoder(args), get_text_encoder(args, tokenizer=text_tokenizer)

    if args.encoder_approach == 'simple_tr_encoder':
        graph_processor = GraphContextTransformerEncoder(visual_model, args.image_emb_size,
                                                         torch.nn.Sequential(text_embedding, textual_model),
                                                         args.text_emb_size, args.gat_feature_size, args.gat_depth,
                                                         args.gat_width,device=args.device,
                                                         freeze_encoder=args.freeze_backbone,
                                                         train_text=args.train_text, train_vision=args.train_vision
                                                         )

    elif args.encoder_approach == 'gat_encoder':

        graph_processor = GraphContextGAT(visual_model, args.image_emb_size, textual_model, args.text_emb_size,
                                          args.gat_feature_size, graph_embedding, args.gat_depth, args.gat_width,
                                          args.gat_in_channels, args.gat_hidden_channels, device=args.device,
                                          freeze_encoder=args.freeze_backbone or args.train_text)
    else:
        raise NotImplementedError(f"Encoder approach {args.encoder_approach} not implemented")

    if args.decoder_architecture == 'tr':
        decoder = TransformerDecoder(graph_processor, args.gat_feature_size, text_embedding,
                                     args.decoder_emb_size, args.decoder_depth, len(text_tokenizer), args.decoder_width,
                                     args.text_context_size, text_tokenizer.eos_token_id, text_tokenizer.bos_token_id,
                                     args.freeze_backbone, args.auto_recurrent_decoder)
    elif args.decoder_architecture == 'tstr':

        special_tokens = [f"[{x}]" for x in text_tokenizer.special_tokens]
        # text_tokenizer.tokenizer is a bert tokenizer

        special_tokens_idx = text_tokenizer.tokenizer.convert_tokens_to_ids(special_tokens)

        decoder = TwoStagesTransformerDecoder(special_tokens_idx, graph_processor, args.gat_feature_size, text_embedding,
                                 args.decoder_emb_size, args.decoder_depth, len(text_tokenizer), args.decoder_width,
                                 args.text_context_size, text_tokenizer.eos_token_id, text_tokenizer.bos_token_id,
                                 args.freeze_backbone, args.auto_recurrent_decoder)
    elif args.decoder_architecture == 'lstm':
        decoder = LSTMDecoderWithAttention(graph_processor, args.gat_feature_size, textual_model.model.token_embedding,
                                           args.decoder_emb_size, len(text_tokenizer),
                                           start_token_id=text_tokenizer.bos_token_id,
                                           max_tokens_during_train=args.text_context_size,
                                           decoder_depth=args.decoder_depth,
                                           decoder_heads=args.decoder_width)
    elif args.decoder_architecture=='gpt2':
        decoder=GPT2Decoder(graph_processor, args.gat_feature_size, args.decoder_emb_size, args.decoder_depth,
                            len(text_tokenizer), args.decoder_width, textual_model.model.token_embedding,
                            max_tokens_during_train=args.text_context_size, stop_token=text_tokenizer.eos_token_id,
                            start_token_id=text_tokenizer.bos_token_id, train_text_emb=False)

    else:
        raise NotImplementedError(f"{args.decoder_architecture} is not an implemented decoder.")
    if args.load_global_checkpoint:

        decoder.load_state_dict(torch.load(args.load_global_checkpoint))

    return decoder, graph_tokenizer, text_tokenizer

def prepare_data(args, text_tokenizer, graph_tokenizer):

    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224), ),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize(IMAGENET_MEANS, IMAGENET_STDS)], )
    collator = Collator(transforms, text_tokenizer, graph_tokenizer, use_sbert=args.use_sbert)
    if args.dataset == 'coco':
        # Preparing train data
        train_dir = os.path.join(args.coco_base_dir, 'train2017')
        train_file = os.path.join(
            args.coco_base_dir, 'annotations', 'captions_train2017.json')
        train_set = CocoCaption(train_dir, read_json(
            train_file), to_text_nodes=args.nodes_to_text)

        # Preparing test data
        val_dir = os.path.join(args.coco_base_dir, 'val2017')
        val_file = os.path.join(
            args.coco_base_dir, 'annotations', 'captions_val2017.json')
        test_set = CocoCaption(val_dir, read_json(
            val_file), to_text_nodes=args.nodes_to_text)

    elif args.dataset == 'gcc':
        train_set = GCCaptions(args.gcc_base_dir, 'training', args.nodes_to_text)
        test_set = GCCaptions(args.gcc_base_dir, 'validation', args.nodes_to_text)

    elif args.dataset=='hmmkg':
        dataset_kwargs = {
            'random_walk_leng': args.random_walk_len,
            'neighbor_context_window': args.context_neight_depth,
            'text_processor_nodes': args.nodes_to_text,
            'masked_captions_csv_path': args.maked_captions_csv
        }


        train_set, test_set = (MaskedCaptionningDataset(**{**dataset_kwargs, **{'split': 'train'}}), # TODO: WARNING! THIS IS SWITCHED S ITS EASY TO OVERFIT
                               MaskedCaptionningDataset(**{**dataset_kwargs, **{'split': 'test'}}))
    else: raise NotImplementedError('Please, provide a valid dataset, mate...')

    if args.encoder_approach == 'simple_tr_encoder':
        collate_fn = collator.simple_encoder_with_adj_collate

    elif args.encoder_approach == 'gat_encoder':
        collate_fn = collator.base_collate_captioning

    else:
        raise NotImplementedError(f"Encoder approach {args.encoder_approach} not implemented")

    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'collate_fn': collate_fn,
        'num_workers': args.num_workers
    }

    return (DataLoader(train_set, **{**dataloader_kwargs, **{'shuffle': True}}),
            DataLoader(test_set, **{**dataloader_kwargs, **{'shuffle': False}}))

def get_tokenizers(args):

    if args.text_tokenizer=='CLIP':
        text_tokenizer = CLIPOriginalTokenizer(clip_model_tag=args.clip_tag, context_length=args.text_context_size)
    elif args.text_tokenizer=='bert':
        text_tokenizer = BERTTokenizer(context_length=args.text_context_size)
    else: raise NotImplementedError(f'{args.text_tokenizer} is not an implemented tokenizer')

    return (GraphTokenizer(args.path_to_gexf_graph, valid_node_categories=args.valid_categories,
                           checkpoint=args.graph_tokenizer_path),
            text_tokenizer)

def get_optimizer(args):
    optim_args = {
        'lr': args.lr
    }
    if args.optimizer == 'Adam':
        return torch.optim.Adam(**optim_args)

    elif args.optimizer == 'AdamW':
        return torch.optim.AdamW(**optim_args)

    elif args.optimizer == 'RMSprop':
        return torch.optim.RMSprop(**optim_args)

    else:
        raise NotImplementedError(f"Not implemented {args.optimizer} optimizer")


def main(args):
    model, graph_tokenizer, text_tokenizer = prepare_models(args)
    train_loader, test_loader = prepare_data(args, text_tokenizer, graph_tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.log_wandb:

        import wandb
        wandb.init(project='hmmkg_captioning')
        wandb.config.update(args)
        logger=wandb

    else: logger = None

    personal_best = np.inf
    loss_args = {'ignore_index': 0}
    if args.use_cross_entropy_weights:
        print('(script) Using Cross Entropy weights')
        loss_args['weight'] = compute_or_get_vocab_weights(train_loader.dataset,
                                                            text_tokenizer,
                                                            0,
                                                            len(text_tokenizer)
                                                            ).to(args.device)
    loss_function = torch.nn.CrossEntropyLoss(**loss_args)
    for epoch in range(args.epoches):

        print(f"-----Training Epoch {epoch}--------")
        train_loss = cross_entropy_train_loop_for_masked_filling(train_loader, optimizer, model, logger=logger, epoch=epoch,
                                              loss_function=loss_function, tokenizer=text_tokenizer)
        print(f"(Script) Trained epoch {epoch} with loss {train_loss}")

        test_metrics = eval(test_loader, model, text_tokenizer, logger=logger,
                            loss_function=loss_function)
        print(f"(Script) Tested epoch {epoch} with metrics {test_metrics}")
        # if personal_best <= train_loss and args.save_checkpoint_to: # TODO: Don't take loss as personal best!!
        torch.save(model.state_dict(), args.save_checkpoint_to)
        #    personal_best = train_loss


if __name__ == '__main__':
    from src._io.args import parse_arguments
    args = parse_arguments()
    main(args)