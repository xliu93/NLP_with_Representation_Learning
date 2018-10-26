import os
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from constants import DataFileName, DATA_PATH


def get_fasttext_embedding(vocab_size, start_idx=0):
    ft_file = DATA_PATH + DataFileName.FT_VOCAB
    with open(ft_file) as f:
        loaded_embeddings_ft = np.zeros((vocab_size, 300))
        words_ft = {}
        idx2words_ft = {}
        ordered_words_ft = []
        # Each line in FastText pre-trained word vectors file:
        # 0-index: word
        # following: embedded vectors
        for i, line in enumerate(f):
            if i >= vocab_size:
                break
            s = line.split()
            loaded_embeddings_ft[i, :] = np.asarray(s[1:])
            words_ft[s[0]] = i + start_idx
            idx2words_ft[i + start_idx] = s[0]
            ordered_words_ft.append(s[0])
    return words_ft, idx2words_ft, loaded_embeddings_ft  # ordered_words_ft ? return it or not?


def compute_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()
    for prem, hypo, p_len, h_len, labels in loader:
        outputs = F.softmax(model(prem, hypo, p_len, h_len), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        # print(predicted)
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return 100 * correct / total


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--d_proj', type=int, default=300)
    parser.add_argument('--d_hidden', type=int, default=300)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
