import torch
from torch.utils.data import Dataset
import numpy as np


class IMDBDatum:
    """
    Class that represents a train/validation/test datum
    - self.raw_text
    - self.label: 0 neg, 1 pos
    - self.file_name: dir for this datum
    - self.tokens: list of tokens
    - self.token_idx: index of each token in the text
    """

    def __init__(self, raw_text, label, file_name):
        self.raw_text = raw_text
        self.label = label
        self.file_name = file_name

    def set_ngram(self, ngram_ctr):
        self.ngram = ngram_ctr

    def set_token_idx(self, token_idx):
        self.token_idx = token_idx

    def set_tokens(self, tokens):
        self.tokens = tokens


class IMDBDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list):
        """
        @param data_list: list of IMDBDatum
        """
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        token_idx, label = self.data_list[key].token_idx, self.data_list[key].label
        return (token_idx, len(token_idx)), label


def imdb_collate_func(batch, pad_index=1):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    for datum in batch:
        label_list.append(datum[1])
        length_list.append(datum[0][1])
    max_length = np.max(length_list)
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0][0]),
                            pad_width=((0, max_length - datum[0][1])),
                            mode="constant", constant_values=pad_index)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), torch.LongTensor(length_list), torch.LongTensor(label_list)]


# # consturct datasets
# imdb_train = IMDBDataset(train_data)
# imdb_validation = IMDBDataset(validation_data)
# imdb_test = IMDBDataset(test_data)
#
# # construct data loader
# train_loader = torch.utils.data.DataLoader(dataset=imdb_train,
#                                            batch_size=batch_size,
#                                            collate_fn=imdb_collate_func,
#                                            shuffle=True)
# validation_loader = torch.utils.data.DataLoader(dataset=imdb_validation,
#                                                 batch_size=batch_size,
#                                                 collate_fn=imdb_collate_func,
#                                                 shuffle=False)
# test_loader = torch.utils.data.DataLoader(dataset=imdb_test,
#                                           batch_size=batch_size,
#                                           collate_fn=imdb_collate_func,
#                                           shuffle=False)
