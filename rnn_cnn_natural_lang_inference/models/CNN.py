import torch
import torch.nn as nn

from constants import HParamKey

DefaultConfig = {
    HParamKey.VOCAB_SIZE: 50000,
    HParamKey.EMB_SIZE: 300,
    HParamKey.HIDDEN_SIZE: 200
}


class Encoder(nn.Module):
    """
    Encoder for CNN Model
    A 2-layer, 1-D convolutional network with ReLU activations
    A max-pool of last hidden representation will be considered as the encoder output.
    """

    def __init__(self, config):
        super(Encoder, self).__init__()
        pass

    def forward(self, inputs):
        pass


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        pass

    def forward(self, batch):
        pass
