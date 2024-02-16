import torch
from tqdm import tqdm
def cross_entropy_train_loop(dataloader, optimizer, model, loss_function = torch.nn.CrossEntropyLoss(), logger = None, epoch=0):

    model.train()
    losses = []
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        output = model(batch)['language_head_output'].transpose(1, 0)
        ouput_flattened = output.reshape(output.shape[0] * output.shape[1], output.shape[-1])
        labels = batch['captions'].reshape(output.shape[0] * output.shape[1]).to(output.device)
        loss = loss_function(ouput_flattened, labels)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())
        if logger:
            logger.log(
                {'batch_loss': loss.item(),
                'epoch': epoch}
            )

    avg_loss = sum(losses) / len(losses)
    if logger:
        logger.log({
            'epoch_loss': avg_loss
        })
    return avg_loss