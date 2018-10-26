import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import HParamKey

DEFAULT_HIDDEN_SIZE = 200
DEFAULT_NUM_LAYERS = 2
NUM_CLASSES = 3
DEFAULT_KERNEL_SIZE = 5


class Encoder(nn.Module):
    """
    Encoder for CNN Model
    A 2-layer, 1-D convolutional network with ReLU activations
    A max-pool of last hidden representation will be considered as the encoder output.
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        pad_size = kernel_size // 2
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding=pad_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=pad_size)

    def forward(self, inputs):
        batch_size, seq_len = inputs.size()[0], inputs.size()[1]
        # 1st layer
        hidden = self.conv1(inputs.transpose(1, 2)).transpose(1, 2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        # 2nd layer
        hidden = self.conv2(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, seq_len, hidden.size(-1))
        # max-pooling
        mp = nn.MaxPool2d((seq_len, 1))
        hidden = mp(hidden).reshape(batch_size, self.hidden_size)
        return hidden


class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()

        self.num_layers = config.get(HParamKey.NUM_LAYER, DEFAULT_NUM_LAYERS)
        self.hidden_size = config.get(HParamKey.HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE)
        self.num_classes = config.get(HParamKey.NUM_CLASS, NUM_CLASSES)
        self.kernel_size = config.get(HParamKey.KERNEL_SIZE, DEFAULT_KERNEL_SIZE)
        # embedding
        trained_emb = torch.from_numpy(config['trained_emb'])  # DoubleTensor
        self.vocab_size, self.emb_size = trained_emb.shape
        self.embed = nn.Embedding.from_pretrained(trained_emb.float())
        # encoder
        self.encoder = Encoder(self.emb_size, self.hidden_size, self.kernel_size)
        # scoring
        self.softmax = nn.Softmax(dim=1)
        # todo: decide in_feature dimension
        self.linear = nn.Linear(self.hidden_size * 2, self.num_classes)

    def forward(self, prem, hypo, p_len, h_len):
        # embedding
        prem_embed = self.embed(prem)
        hypo_embed = self.embed(hypo)
        # encoding
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        # concat
        cat_encoded = torch.cat([premise, hypothesis], dim=1)
        # scoring
        scores = self.softmax(cat_encoded)
        scores = self.linear(scores)
        return scores
