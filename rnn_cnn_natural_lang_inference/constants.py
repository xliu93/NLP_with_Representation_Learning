"""
Constants and Keys
"""
import torch
import sys

# Dynamic data path
if sys.platform == 'linux':
    DATA_PATH = '/scratch/xl2053/nlp/hw2_data/'  # NYU HPC
elif sys.platform == 'darwin':
    DATA_PATH = '/Users/xliu/Downloads/hw2_data/'  # local

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 200
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
PAD_IDX = 0
UNK_IDX = 1


class FTcorpus:
    WIKI_NEWS = 'news'
    COMM_CRAWL = 'cc'


class DataFileName:
    MNLI_TRAIN = 'mnli_train.tsv'
    MNLI_VAL = 'mnli_val.tsv'
    SNLI_TRAIN = 'snli_train.tsv'
    SNLI_TRAIN_MINI = 'snli_train_mini.tsv'  # a subset of SNLI train set
    SNLI_VAL = 'snli_val.tsv'
    FT_NEWS_VOCAB = 'wiki-news-300d-1M.vec'
    FT_CC_VOCAB = 'crawl-300d-2M.vec'


class TrainRecordKey:
    TRAIN_ACC = 'train_accuracy'
    VAL_ACC = 'val_accuracy'
    TRAIN_LOSS = 'train_loss'
    VAL_LOSS = 'val_loss'


class HParamKey:
    # vocabulary
    FT_CORPUS_NAME = 'ft_corpus_name'
    VOCAB_SIZE = 'vocab_size'
    EMB_SIZE = 'emb_size'
    # model
    HIDDEN_SIZE = 'hidden_size'
    NUM_LAYER = 'num_layers'
    NUM_CLASS = 'num_classes'
    KERNEL_SIZE = 'kernel_size'
    DROPOUT_PROB = 'dropout_prob'
    WEIGHT_DECAY = 'weight_decay'
    # train
    BATCH_SIZE = 'batch_size'
    NUM_EPOCHS = 'num_epochs'
    LEARNING_RATE = 'learning_rate'
    LR_DECAY = 'lr_decay'
    IF_EARLY_STOP = 'if_earlystop'
    ES_LOOKBACK = 'earlystop_lookback'
    MODEL_SAVE_PATH = 'model_save_path'
    # no tuning for optimizer in this task


DefaultConfig = {
    HParamKey.FT_CORPUS_NAME: 'news',
    HParamKey.VOCAB_SIZE: 100000,
    HParamKey.NUM_CLASS: 3,
    HParamKey.NUM_LAYER: 1,
    HParamKey.HIDDEN_SIZE: 100,
    HParamKey.KERNEL_SIZE: 3,
    HParamKey.WEIGHT_DECAY: 0.0,  # no weight decay (L2) by default
    HParamKey.DROPOUT_PROB: 0.0,  # no dropout by default
    HParamKey.NUM_EPOCHS: 10,
    HParamKey.BATCH_SIZE: 256,
    HParamKey.LEARNING_RATE: 0.01,
    HParamKey.LR_DECAY: 0.0,     # 0.0 for no decay
    HParamKey.IF_EARLY_STOP: False,
    HParamKey.ES_LOOKBACK: 5,
    HParamKey.MODEL_SAVE_PATH: './'
}


LogConfig = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] [%(levelname)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'filters': {},
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'default': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': None,  # to be override
            'formatter': 'standard',
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'default'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
