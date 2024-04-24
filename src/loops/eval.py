from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
import torch
def eval(dataloader, model, tokenizer, loss_function = torch.nn.CrossEntropyLoss(), logger=None):

    rouge_score = ROUGEScore()
    bleu_scorer = BLEUScore()

    with ((torch.no_grad())):
        avg_bleu, avg_perplex, avg_rouge, avg_bert, avg_loss = 0,0,0,0,0
        avg_bleu_unmasked, avg_perplex_unmasked, avg_rouge_unmasked, avg_bert_unmasked, avg_loss_unmasked = \
            0, 0, 0, 0, 0
        for num, batch in tqdm(enumerate(dataloader)):

            decoded_out = model.eval_forward(batch)

            decoded_labels = [tokenizer.decode(row[:1 + row.index(tokenizer.eos_token_id)])
                              for row in batch['captions'].cpu().numpy().tolist()]


            decoded_prediction = [tokenizer.decode(row[:(row.index(tokenizer.eos_token_id)+1
                                                         if tokenizer.eos_token_id in row else -1)])
                              for row in decoded_out['generated_sequence'].cpu().numpy().tolist()]

            avg_rouge += rouge_score(decoded_prediction, decoded_labels)['rouge1_fmeasure'].item()
            avg_bleu += bleu_scorer(decoded_prediction, [[x] for x in decoded_labels]).item()

            if 'masked_generated_sequence' in decoded_out:
                decoded_labels_unmasked = [tokenizer.decode(row[:1 + row.index(tokenizer.eos_token_id)])
                                  for row in batch['unmasked_captions'].cpu().numpy().tolist()]
                decoded_prediction_unmkased = [tokenizer.decode(row[:(row.index(tokenizer.eos_token_id) + 1
                                                             if tokenizer.eos_token_id in row else -1)])
                                      for row in decoded_out['masked_generated_sequence'].cpu().numpy().tolist()]

                avg_rouge_unmasked += rouge_score(decoded_prediction_unmkased, decoded_labels_unmasked)['rouge1_fmeasure'].item()
                avg_bleu_unmasked += bleu_scorer(decoded_prediction_unmkased, [[x] for x in decoded_labels_unmasked]).item()

    print('\n'.join([f'\n------------------\n\nPredicted \ actual:\n\tMasked Prediction: {x}, \n\tUnmasked Prediction: '
                     f'{z} \n- gt -\n \n\tMasked GT: {y}, \n\tUnmasked GT: {w}\nNodes: {list(g.nodes())}' for x,y,z,w,g in zip(decoded_prediction,
                                                                         decoded_labels, decoded_prediction_unmkased,
                                                                             decoded_labels_unmasked, batch['graph']['graphs'])]))


    res_dict = {
        #'avg_loss': avg_loss/(num + 1),
        'avg_rouge': avg_rouge/(num+1),
        'avg_bleu': avg_bleu/(num+1),
        'avg_rouge_unmasked': avg_rouge_unmasked / (num + 1),
        'avg_bleu_unmasked': avg_bleu_unmasked / (num + 1),

    }
    if logger:
        logger.log(res_dict)
    return res_dict



