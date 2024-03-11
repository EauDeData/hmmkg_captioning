import torch
from tqdm import tqdm
def cross_entropy_train_loop(dataloader, optimizer, model, loss_function = torch.nn.CrossEntropyLoss(), logger = None,
                             epoch=0, tokenizer=lambda x: x):

    model.train()
    losses = []
    for batch in tqdm(dataloader):

        optimizer.zero_grad()
        decoded_out = model.forward(batch)
        output = decoded_out['language_head_output'].transpose(1, 0) # --> (BS, SEQ, VOCAB)
        ouput_flattened = output.reshape(output.shape[0] * output.shape[1], output.shape[-1])

        # batch['captions']: (BS, SEQ)
        labels = batch['captions'].reshape(output.shape[0] * output.shape[1]).to(output.device)
        loss = loss_function(ouput_flattened, labels)
        loss.backward()

        # print(loss)

        optimizer.step()

        losses.append(loss.item())

        if logger:
            logger.log(
                {
                'batch_loss': loss.item(),
                'epoch': epoch
                 }
            )

    avg_loss = sum(losses) / len(losses)
    if logger:
        logger.log({
            'epoch_loss': avg_loss
        })


    print('---------train example---------------')
    with torch.no_grad():
        decoded_eval = model.eval_forward(batch)

    decoded_labels = [tokenizer.decode(row[:1 + row.index(tokenizer.eos_token_id)])
                      for row in batch['captions'].cpu().numpy().tolist()]

    decoded_prediction = [tokenizer.decode(row[:(row.index(tokenizer.eos_token_id) + 1
                                                 if tokenizer.eos_token_id in row else -1)])
                          for row in decoded_out['generated_sequence'].cpu().numpy().tolist()]

    decoded_prediction_autoreg = [tokenizer.decode(row[:(row.index(tokenizer.eos_token_id) + 1
                                                 if tokenizer.eos_token_id in row else -1)])
                          for row in decoded_eval['generated_sequence'].cpu().numpy().tolist()]

    print('\n'.join([f'Predicted \ actual \ decoded_eval: \n\t{x}, \n\t| {y} |\n\t {z}' for x,y,z in zip(decoded_prediction,
                                                                         decoded_labels,decoded_prediction_autoreg)
                     ]))

    print('-----------------------------------')

    return avg_loss