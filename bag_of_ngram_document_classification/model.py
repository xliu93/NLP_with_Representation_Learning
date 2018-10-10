import torch
import torch.nn as nn


class FastText(nn.Module):
    """
    FastText model
    """

    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary.
        @param emb_dim: size of the word embedding
        """
        super(FastText, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 2, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim, 1)

    def forward(self, data, length):
        """
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        batch_size = data.size()[0]
        out = self.embedding(data)  # the result is (batch_size * max_len * emb_dim)
        out = out.sum(dim=1)  # (batch_size * emb_dim)
        out = out.div(length.type(torch.FloatTensor).view(batch_size, 1))  # averaging
        out = self.linear(out)  # feed to the linear classifier
        return torch.sigmoid(out.view(-1))

