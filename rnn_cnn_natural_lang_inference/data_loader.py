import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import (DATA_PATH, DataFileName, DEVICE,
                       MAX_LENGTH, PAD_IDX, UNK_IDX)

LABEL_IDX = [
    'entailment',
    'neutral',
    'contradiction'
]


def get_snli_data():
    """
    Get the language data from file
    :return:
    train_data: pandas.DataFrame with columns ['sentence1', 'sentence2', 'label']
    valid_data: pandas.DataFrame with same columns as above
    """
    train_file = DATA_PATH + DataFileName.SNLI_TRAIN  # pre-train test
    val_file = DATA_PATH + DataFileName.SNLI_VAL
    train_data = pd.read_csv(train_file, sep='\t')
    valid_data = pd.read_csv(val_file, sep='\t')
    return train_data, valid_data


def get_mnli_data():
    train_file = DATA_PATH + DataFileName.MNLI_TRAIN
    val_file = DATA_PATH + DataFileName.MNLI_VAL
    train_data = pd.read_csv(train_file, sep='\t')
    val_data = pd.read_csv(val_file, sep='\t')
    return train_data, val_data


def token2idx_dataset(sentence_list, token2idx):
    def sentence2idx(sentence):
        # clean-up (?) FastText contains both upper-case and lower-case tokens
        # sentence = sentence.lower().replace('\n', '')
        # convert token to index
        return [token2idx[token] if token in token2idx else UNK_IDX for token in sentence.split(' ')]

    indices_data = list(map(sentence2idx, sentence_list))
    return indices_data


def snli_token2id(data_set, indexer):
    sent1 = data_set['sentence1'].values
    sent2 = data_set['sentence2'].values
    label = data_set['label'].values
    sent1_idx = token2idx_dataset(sent1, indexer)
    sent2_idx = token2idx_dataset(sent2, indexer)
    label = list(map(lambda i: LABEL_IDX.index(i), label))
    return sent1_idx, sent2_idx, label


def snli_collate_func(batch):
    prem_list = []
    prem_len_list = []
    hypo_list = []
    hypo_len_list = []
    label_list = []
    for datum in batch:
        # 0: prem, 1: hypo, 2: prem_len, 3: hypo_len, 4: label
        label_list.append(datum[-1])
        prem_len_list.append(datum[2])
        hypo_len_list.append(datum[3])
        # padding
        padded_prem = np.pad(np.array(datum[0]),
                             pad_width=((0, MAX_LENGTH-datum[2])),
                             mode='constant',
                             constant_values=PAD_IDX)
        prem_list.append(padded_prem)
        padded_hypo = np.pad(np.array(datum[1]),
                             pad_width=((0, MAX_LENGTH - datum[3])),
                             mode='constant',
                             constant_values=PAD_IDX)
        hypo_list.append(padded_hypo)
    return [torch.from_numpy(np.array(prem_list)).to(DEVICE),
            torch.from_numpy(np.array(hypo_list)).to(DEVICE),
            torch.LongTensor(prem_len_list).to(DEVICE),
            torch.LongTensor(hypo_len_list).to(DEVICE),
            torch.LongTensor(label_list).to(DEVICE)]


class snliDataset(Dataset):
    """
    Class that represents a train/validation dataset of SNLI that is
    readable for Pytorch, which inherits torch.utils.data.Dataset.
    """
    def __init__(self, prem_list, hypo_list, label_list):
        """

        :param data_list:
        :param label_list:
        """
        self.prem_list = prem_list
        self.hypo_list = hypo_list
        self.label_list = label_list

    def __len__(self):
        return len(self.prem_list)

    def __getitem__(self, key):
        prem_idx = self.prem_list[key][:MAX_LENGTH]
        hypo_idx = self.hypo_list[key][:MAX_LENGTH]
        label = self.label_list[key]
        return [prem_idx, hypo_idx, len(prem_idx), len(hypo_idx), label]
