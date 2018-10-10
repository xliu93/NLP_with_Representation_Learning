from torch.utils.data import Dataset
import torch.nn as nn
import logging
import logging.config
import pickle as pkl
import time

from constants import LogConfig
from structures import IMDBDataset
from model import FastText
from ngram_extraction import *
from utils import *

# config logger
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger()
LogConfig['handlers']['default']['filename'] = './log/FT-Ngram {}'.format(time.time())
logging.config.dictConfig(LogConfig)
logger.info('>>> Document Classification with Bag of N-gram embeddings. >>>')

# I/O Param
data_dir = "/Users/xliu/Documents/NLP_with_representation/homework/hw01/aclImdb/"
# data_dir = "./aclImdb/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
TRAIN_SIZE = 20000
VALIDATION_SIZE = 5000
TEST_SIZE = 25000
PADDING_IDX = 1

###################
# Hyperparameters #
###################

# vocab_size = [5000, 10000, 20000, 50000, 100000, 200000]
# ns = [1, 2, 3, 4]
vocab_size = [5000]
ns = [1]

################
# Data Loading #
################
logger.info('--- load data...')
# Read data and split train/validation
train_set = construct_dataset(train_dir, TRAIN_SIZE)
logger.info('--- train set loaded')
validation_set = construct_dataset(train_dir, VALIDATION_SIZE, offset=int(TRAIN_SIZE/2))
logger.info('--- validation set loaded')
test_set = construct_dataset(test_dir, TEST_SIZE)
logger.info('--- test set loaded')

for v in vocab_size:
    for n in ns:
        logger.info("=== processing: N={}, vocab_size={}".format(n, v))
        train = train_set.copy()
        val = validation_set.copy()
        test = test_set.copy()
        train_data, train_ngram_indexer = process_text_dataset(train, n, topk=v, use_spacy=True)
        validation_data, _ = process_text_dataset(val, n, ngram_indexer=train_ngram_indexer, use_spacy=True)
        test_data, _ = process_text_dataset(test, n, ngram_indexer=train_ngram_indexer, use_spacy=True)
        spec = 'ngram_{}_{}_spacy'.format(n, v)
        pkl.dump(train_data, open("./data/train_{}.p".format(spec), "wb"))
        pkl.dump(validation_data, open("./data/valid_{}.p".format(spec), "wb"))
        pkl.dump(test_data, open("./data/test_{}.p".format(spec), "wb"))
        pkl.dump(train_ngram_indexer, open("./data/indexer_{}.p".format(spec), "wb"))
logger.info('--- Task completed!')

