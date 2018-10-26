import torch
import torch.nn as nn

from constants import HParamKey

DEFAULT_HIDDEN_SIZE = 200
DEFAULT_NUM_LAYERS = 1
NUM_CLASSES = 3


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          bidirectional=True,
                          batch_first=True)

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        state_shape = 2, batch_size, self.hidden_size  # 2 for bi-directional,
        h0 = inputs.new_zeros(state_shape)
        outputs, hidden_state = self.rnn(inputs, h0)
        # print("in Encoder: hidden_state", hidden_state.shape)
        o = hidden_state[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        # print("operate on last hidden state:", o.shape)
        return o


class RNNModel(nn.Module):
    def __init__(self, config):
        """

        :param config:
        {
            'num_layers': number of layers of RNN in encoder
            'hidden_size': hidden size of neural network
            'num_classes': number of classes for prediction
            'pre_trained_emb': matrix of pre-trained word vectors, size (vocab_size, emb_size)
        }
        """
        super(RNNModel, self).__init__()
        self.num_layers = config.get(HParamKey.NUM_LAYER, DEFAULT_NUM_LAYERS)
        self.hidden_size = config.get(HParamKey.HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE)
        self.num_classes = config.get(HParamKey.NUM_CLASS, NUM_CLASSES)
        # embedding
        trained_emb = torch.from_numpy(config['trained_emb'])  # DoubleTensor
        self.vocab_size, self.emb_size = trained_emb.shape
        self.embed = nn.Embedding.from_pretrained(trained_emb.float())
        # encoder
        self.encoder = Encoder(self.emb_size, self.hidden_size)
        # scoring
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(in_features=self.hidden_size*4, out_features=self.num_classes)

    def forward(self, prem, hypo, p_len, h_len):
        # todo: confirm and remove length variables
        prem_embed = self.embed(prem)
        hypo_embed = self.embed(hypo)
        # print("prem, hypo embedded:", prem_embed.shape, hypo_embed.shape)
        # (batch * max_len * embedding)
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)
        prem_hypo = torch.cat([premise, hypothesis], dim=1)
        # print("after cat:", prem_hypo.shape)
        scores = self.softmax(prem_hypo)
        # print(scores.shape)
        scores = self.linear(scores)
        # print(scores.shape)
        return scores