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
        # Displace it one step, so I have to predict the next one.
        labels = torch.cat((batch['captions'][:, 1:], torch.zeros_like(batch['captions'][:, :1])), dim=1)\
            .reshape(output.shape[0] * output.shape[1]).to(output.device)
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

    return avg_loss
