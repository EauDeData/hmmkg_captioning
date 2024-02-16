from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from tqdm import tqdm
import torch
def eval(dataloader, model, tokenizer, loss_function = torch.nn.CrossEntropyLoss(), logger=None):

    rouge_score = ROUGEScore()
    bleu_scorer = BLEUScore()

    with torch.no_grad():
        avg_bleu, avg_perplex, avg_rouge, avg_bert, avg_loss = 0,0,0,0,0
        for num, batch in tqdm(enumerate(dataloader)):

            decoded_out = model.forward(batch) # TODO: This is just a quick debugging,\
            #TODO: WARNING!! use eval_forward for fair comparisons
            output =decoded_out['language_head_output'].transpose(1, 0)
            ouput_flattened = output.reshape(-1, output.shape[-1])
            labels = batch['captions'].view(-1).to(output.device)
            loss = loss_function(ouput_flattened, labels)

            decoded_labels = [tokenizer.decode(row[:1 + row.index(tokenizer.eos_token_id)])
                              for row in batch['captions'].cpu().numpy().tolist()]


            decoded_prediction = [tokenizer.decode(row[:(row.index(tokenizer.eos_token_id)+1
                                                         if tokenizer.eos_token_id in row else -1)])
                              for row in decoded_out['generated_sequence'].cpu().numpy().tolist()]

            avg_loss += loss.item()
            avg_rouge += rouge_score(decoded_prediction, decoded_labels)['rouge1_fmeasure'].item()
            avg_bleu += bleu_scorer(decoded_prediction, [[x] for x in decoded_labels]).item()

    print('\n'.join([f'Predicted \ actual:\n\t{x.replace("!", "")}, | {y}' for x,y in zip(decoded_prediction,
                                                                         decoded_labels)]))

    res_dict = {
        'avg_loss': avg_loss/(num + 1),
        'avg_rouge': avg_rouge/(num+1),
        'avg_bleu': avg_bleu/(num+1)
    }
    if logger:
        logger.log(res_dict)
    return res_dict



