import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import logging.config

from constants import (DataFileName, DATA_PATH, LogConfig, TrainRecordKey,
                       PAD_TOKEN, PAD_IDX, UNK_TOKEN, UNK_IDX)


def init_logger(logfile=None, loglevel=logging.INFO):
    logging.getLogger('__main__').setLevel(loglevel)
    if logfile is None:
        LogConfig['loggers']['']['handlers'] = ['console']
        LogConfig['handlers']['default']['filename'] = 'demo.log'
    else:
        LogConfig['loggers']['']['handlers'] = ['console', 'default']
        LogConfig['handlers']['default']['filename'] = logfile
    logging.config.dictConfig(LogConfig)


def get_results_folder(base_path='./'):
    fd_name = '{}results_{}/'.format(base_path, int(time.time()))
    os.makedirs(fd_name, exist_ok=True)
    os.makedirs(fd_name+'checkpoints/', exist_ok=True)
    return fd_name


def get_fasttext_embedding(vocab_size, corpus_name='news', start_idx=2):
    """

    :param vocab_size: number of words to select
    :param corpus_name: for FastText, use Wiki-News or Common Crawl corpus, {'news', 'cc'}
    :param start_idx: the start index instead of 0 when working with PAD_IDX and UNK_IDX
    :return:
    """
    # locate file
    ft_file = DATA_PATH + (DataFileName.FT_NEWS_VOCAB if corpus_name == 'news' else DataFileName.FT_CC_VOCAB)

    # read file and build vocabulary
    loaded_embeddings_ft = np.zeros((vocab_size, 300))
    words_ft = {PAD_TOKEN: PAD_IDX,
                UNK_TOKEN: UNK_IDX}
    idx2words_ft = {PAD_IDX: PAD_TOKEN,
                    UNK_IDX: UNK_TOKEN}
    ordered_words_ft = [PAD_TOKEN, UNK_TOKEN]
    loaded_embeddings_ft[PAD_IDX, :] = np.zeros((1, 300))
    loaded_embeddings_ft[UNK_IDX, :] = np.random.rand(1, 300)
    with open(ft_file) as f:
        # Each line in FastText pre-trained word vectors file:
        # 0-index: word
        # following: embedded vectors
        for i, line in enumerate(f):
            if i >= (vocab_size - 2):
                break
            s = line.split()
            loaded_embeddings_ft[i + start_idx, :] = np.asarray(s[1:])
            words_ft[s[0]] = i + start_idx
            idx2words_ft[i + start_idx] = s[0]
            ordered_words_ft.append(s[0])
    return words_ft, idx2words_ft, loaded_embeddings_ft


def back_to_sentence(id2token, tsor_ndarray):
    token_list = [id2token[idx] for idx in tsor_ndarray]
    sentence = list(filter(lambda t: t != '<pad>', token_list))
    sentence = ' '.join(sentence)
    return sentence


def record_to_graph(csv_fname):
    hist_df = pd.read_csv(csv_fname, index_col=0)
    train_acc = hist_df[TrainRecordKey.TRAIN_ACC].values
    train_loss = hist_df[TrainRecordKey.TRAIN_LOSS].values
    val_acc = hist_df[TrainRecordKey.VAL_ACC].values
    val_loss = hist_df[TrainRecordKey.VAL_LOSS].values
    # accuracy curves
    f, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(train_acc, label='training acc')
    ax[0].plot(val_acc, label='val acc')
    ax[0].set_xlabel('number of steps * interval')
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].legend()
    # loss curves
    ax[1].plot(train_loss, label='training loss')
    ax2 = ax[1].twinx()
    ax2.plot(val_loss, label='val loss', color='red')
    ax[1].legend()
    ax2.legend(bbox_to_anchor=(1, 0.95))
    ax[1].set_xlabel('number of steps * interval')
    ax[1].set_ylabel('CrossEntropyLoss')
    # save figure
    fig_path = csv_fname.replace('.csv', '.png')
    plt.savefig(fig_path, dpi=300)
