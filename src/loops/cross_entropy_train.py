import torch
from tqdm import tqdm
def cross_entropy_train_loop(dataloader, optimizer, model, loss = torch.nn.CrossEntropyLoss()):

    model.train()
    for batch in tqdm(dataloader):
        output = model(batch)['language_head_output']
        labels = batch['captions']

        print(labels.shape, output.shape)
        exit()