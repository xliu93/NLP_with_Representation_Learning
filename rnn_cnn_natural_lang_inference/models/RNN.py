import torch
import torch.nn as nn

DEFAULT_HIDDEN_SIZE = 200
DEFAULT_NUM_LAYERS = 1
DEFAULT_NUM_CLASSES = 3
DEFAULT_DROPOUT = 0.2


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        if num_layers == 1:
            dropout_prob = 0.0
        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=True,
                          batch_first=True,
                          dropout=dropout_prob)

    def forward(self, inputs):
        batch_size = inputs.size()[0]
        state_shape = 2, batch_size, self.hidden_size  # 2 for bi-directional,
        h0 = inputs.new_zeros(state_shape)
        outputs, hidden_state = self.rnn(inputs, h0)
        o = hidden_state[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        return o


class RNNModel(nn.Module):
    def __init__(self, hidden_size, num_classes, num_layers, dropout_p, trained_emb):
        super(RNNModel, self).__init__()
        # parse parameters
        # self.num_layers = config.get(HParamKey.NUM_LAYER, DEFAULT_NUM_LAYERS)
        # self.hidden_size = config.get(HParamKey.HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE)
        # self.num_classes = config.get(HParamKey.NUM_CLASS, DEFAULT_NUM_CLASSES)
        # self.dropout_rate = config.get(HParamKey.DROPOUT_PROB, DEFAULT_DROPOUT)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout_rate = dropout_p
        # embedding
        trained_emb = torch.from_numpy(trained_emb)  # DoubleTensor
        self.vocab_size, self.emb_size = trained_emb.shape
        self.embed = nn.Embedding.from_pretrained(trained_emb.float())
        # encoder
        self.encoder = Encoder(self.emb_size, self.hidden_size, self.num_layers, self.dropout_rate)
        # scoring
        self.scoring = nn.Sequential(
            nn.Linear(in_features=self.hidden_size * 4, out_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(in_features=self.hidden_size, out_features=self.num_classes)
        )
        # self.linear = nn.Linear(in_features=self.hidden_size*4, out_features=self.num_classes)

    def forward(self, prem, hypo, p_len, h_len):
        # embedding
        prem_embed = self.embed(prem)
        hypo_embed = self.embed(hypo)  # (batch * max_len * embedding)

        # encoding
        premise = self.encoder(prem_embed)
        hypothesis = self.encoder(hypo_embed)

        # concat encoded premise and hypothesis
        prem_hypo = torch.cat([premise, hypothesis], dim=1)
        # print("after cat:", prem_hypo.shape)

        # scoring
        scores = self.scoring(prem_hypo)
        # print(scores.shape)
        return scores
