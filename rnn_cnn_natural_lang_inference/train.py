import os
import time
import glob

import torch
import torch.optim as opt
import torch.nn as nn

from constants import HParamKey
from util import (get_fasttext_embedding, compute_accuracy,
                  makedirs, get_args)
from data_loader import get_snli_data, snli_token2id, snliDataset, snli_collate_func
from models.RNN import RNNModel


# Load pre-trained embeddings of FastText #
word2idx, idx2word, ft_embs = get_fasttext_embedding(50000)

# Load train/validation sets
train_set, val_set = get_snli_data()

# Convert to indices based on FastText vocabulary
train_prem, train_hypo, train_label = snli_token2id(train_set, word2idx)
val_prem, val_hypo, val_label = snli_token2id(val_set, word2idx)

# Create DataLoader
BATCH_SIZE = 32
train_dataset = snliDataset(train_prem, train_hypo, train_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=snli_collate_func,
                                           shuffle=True)
val_dataset = snliDataset(val_prem, val_hypo, val_label)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=BATCH_SIZE,
                                         collate_fn=snli_collate_func,
                                         shuffle=True)

# Get an instance of RNN Model
model_config = {
    HParamKey.HIDDEN_SIZE: 150,
    HParamKey.NUM_LAYER: 1,
    'trained_emb': ft_embs
}
model = RNNModel(config=model_config)

# Initialization for training
learning_rate = 0.01
num_epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = opt.Adam(model.parameters(), lr=learning_rate)

# Train loop
for epoch in range(num_epochs):
    for i, (prem, hypo, p_len, h_len, label) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        outputs = model(prem, hypo, p_len, h_len)  # input for model.forward()
#         print(label.shape)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        if i > 0 and i % 10 == 0:
            train_acc = compute_accuracy(train_loader, model)
            val_acc = compute_accuracy(val_loader, model)
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}, TrainAcc: {} ValAcc: {}'.format(
                   epoch+1, num_epochs, i+1, len(train_loader), loss.item(), train_acc, val_acc))


