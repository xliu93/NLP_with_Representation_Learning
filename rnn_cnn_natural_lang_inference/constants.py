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


class HParamKey:
    VOCAB_SIZE = 'vocab_size'
    EMB_SIZE = 'emb_size'
    HIDDEN_SIZE = 'hidden_size'
    NUM_LAYER = 'num_layers'
    NUM_CLASS = 'num_classes'
    KERNEL_SIZE = 'kernel_size'
    DROPOUT_PROB = 'dropout_prob'


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
