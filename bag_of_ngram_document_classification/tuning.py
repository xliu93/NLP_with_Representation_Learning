# Caution:
# Only use part of the codes for tuning certain hyperparameters.

import logging
import logging.config

from constants import LogConfig
from train import *
from utils import *

###################
# Initialization  #
###################

# config logger
logging.getLogger("__main__").setLevel(logging.INFO)
logger = logging.getLogger()
LogConfig['handlers']['default']['filename'] = 'FT-Ngram'
logging.config.dictConfig(LogConfig)
logger.info('<<< Document Classification with Bag of N-gram embeddings >>>')

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

# number of words in the vocabulary
vocab_size_set = [5000, 10000, 20000, 50000, 100000, 200000]
# embedding size
emb_dim_set = [20, 50, 100, 200, 500, 1000, 1500]
# N-grams
ngram_n_set = [1, 2, 3, 4]
# tokenization
use_spacy = False

# For training:
num_epochs = 20  # number epoch to train
batch_size = 32
learning_rate_set = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
opt_method_set = ['adam', 'sgd']  # all method names in lower case


################
# Data Loading #
################

#========= Method 1. Read raw data and construct data-set

# Read data and split train/validation
# train_set = construct_dataset(train_dir, TRAIN_SIZE)
# validation_set = construct_dataset(train_dir, VALIDATION_SIZE, offset=int(TRAIN_SIZE/2))
# test_set = construct_dataset(test_dir, TEST_SIZE)

# Convert text data into list of index based on N-grams
# train_data, train_ngram_indexer = process_text_dataset(train_set, ngram_n, topk=vocab_size)
# logger.info("train_data ready...")
# validation_data, _ = process_text_dataset(validation_set, ngram_n, ngram_indexer=train_ngram_indexer)
# logger.info("validation set ready...")
# test_data, _ = process_text_dataset(test_set, ngram_n, ngram_indexer=train_ngram_indexer)
# logger.info("test set ready...")

#========= Method 2. Load from preprocessed data files in each training


########################
# Tokenization scheme  #
########################
# Since I only compared using spacy or not (simple split),
# I use a simple switch/flag for this comparison
token_scheme = 'spacy' if use_spacy else 'simple'

#####################################
# Find optimal (N-gram, vocab_size) #
#####################################
best_acc = 0
best_config = {}
for ngram_n in ngram_n_set:
    for vocab_size in vocab_size_set:
        configs = {
            'ngram_n': ngram_n,
            'vocab_size': vocab_size,
            'token_scheme': token_scheme
        }
        model, train_loss, train_acc, valid_acc = train(logger, configs, verbose=False)
        logger.info("Train result: Train Acc:{}%, Validation Acc:{}%, Best Validation Acc:{}".format(
            train_acc[-1], valid_acc[-1], min(valid_acc)))
        if valid_acc[-1] > best_acc:
            best_acc = valid_acc[-1]
            best_config.update(configs)
logger.info("Optimal (N, vocab_size) with {}: ({},{})".format(
    best_config.get('token_scheme'),
    best_config.get('ngram_n'),
    best_config.get('vocab_size')
))


########################
# Find optimal emb_dim #
########################
# Based on previous optimal (N, vocab_size) pair
best_acc = 0
best_config = {}
for emb_dim in emb_dim_set:
    configs = {
        'emb_dim': emb_dim,
        'ngram_n': 4,
        'vocab_size': 100000,
        'token_scheme': token_scheme
    }
    model, train_loss, train_acc, valid_acc = train(logger, configs, verbose=False)
    logger.info("Train result: Train Acc:{}%, Validation Acc:{}%, Best Validation Acc:{}".format(
        train_acc[-1], valid_acc[-1], min(valid_acc)))
    if valid_acc[-1] > best_acc:
        best_acc = valid_acc[-1]
        best_config.update(configs)
logger.info("Optimal emb_dim = {}".format(best_config.get('emb_dim')))


#################################
# Find optimal (opt_method, lr) #
#################################
best_acc = 0
best_config = {}
for opt_method in opt_method_set:
    for lr in learning_rate_set:
        configs = {
            'ngram_n': 4,
            'vocab_size': 100000,
            'token_scheme': token_scheme,
            'emb_dim': 100,
            'learning_rate': lr,
            'opt_method': opt_method
        }
        model, train_loss, train_acc, valid_acc = train(logger, configs, verbose=False)
        logger.info("Train result: Train Acc:{}%, Validation Acc:{}%, Best Validation Acc:{}".format(
            train_acc[-1], valid_acc[-1], min(valid_acc)))
        if valid_acc[-1] > best_acc:
            best_acc = valid_acc[-1]
            best_config.update(configs)
logger.info("Optimal (optim, learning_rate): ({},{})\nFull configs:{}".format(
    best_config.get('opt_method'),
    best_config.get('learning_rate'),
    best_config
))


################################
# Test learning_rate annealing #
################################
lr_annealing_set = [0.5, 0.6, 0.7, 0.8, 0.9, 0]  # 0 for no annealing
best_acc = 0
best_config = {}
for lr_annealing in lr_annealing_set:
    configs = {
        'emb_dim': emb_dim,
        'ngram_n': 4,
        'vocab_size': 100000,
        'token_scheme': token_scheme,
        'opt_method': 'adam',
        'learning_rate': 0.001,
        'lr_annealing': lr_annealing
    }
    model, train_loss, train_acc, valid_acc = train(logger, configs, verbose=False)
    logger.info("Train result: Train Acc:{}%, Validation Acc:{}%, Best Validation Acc:{}".format(
        train_acc[-1], valid_acc[-1], min(valid_acc)))
    if valid_acc[-1] > best_acc:
        best_acc = valid_acc[-1]
        best_config.update(configs)
logger.info("Optimal learning_rate annealing scheme: gamma={}\nFull configs:{}".format(
    best_config.get('lr_annealing'), best_config))


####################
# Test Final Model #
####################
configs = {
    'ngram_n': 4,
    'vocab_size': 100000,
    'emb_dim': 100,
    'token_scheme': 'spacy',
    'learning_rate': 0.001,
    'lr_annealing': 0.9,
    'opt_method': 'adam'
}
model, train_loss, train_acc, valid_acc = train(logger, configs, verbose=True, test_model=True)

