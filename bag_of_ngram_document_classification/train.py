import pickle as pkl
import torch.nn as nn
from torch.autograd import Variable

from model import FastText
from structures import IMDBDataset
from utils import *


def compute_accu(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = Variable(data), Variable(lengths), Variable(labels)
        outputs = model(data_batch, length_batch)
        predicted = (outputs.data > 0.5).long().view(-1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    return 100.0 * correct / total


def earily_stop(val_acc_history, t=2, required_progress=0.01):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    if len(val_acc_history) > t:
        return sum(np.diff(val_acc_history[-t:]) < required_progress) >= t
    return False


def train(logger, configs={}, verbose=True, save_model=False, test_model=False):
    logger.info("Now train a model with configs:\n", configs)
    vocab_size = configs.get('vocab_size', 20000)
    emb_dim = configs.get('emb_dim', 100)
    ngram_n = configs.get('ngram_n', 2)
    batch_size = configs.get('batch_size', 32)
    num_epochs = configs.get('num_epochs', 20)
    token_scheme = configs.get('token_scheme', 'simple')
    learning_rate = configs.get('learning_rate', 0.001)
    lr_annealing = configs.get('lr_annealing', 0)
    opt_method = configs.get('opt_method', 'adam')

    spec = 'ngram_{}_{}_{}'.format(ngram_n, vocab_size, token_scheme)
    train_data = pkl.load(open('./processed/train_{}.p'.format(spec), "rb"))
    # train_ngram_indexer = pkl.load(open('./processed/indexer_{}.p'.format(spec), "rb"))
    validation_data = pkl.load(open('./processed/valid_{}.p'.format(spec), "rb"))
    test_data = pkl.load(open('./processed/test_{}.p'.format(spec), "rb"))

    # consturct datasets
    imdb_train = IMDBDataset(train_data)
    imdb_validation = IMDBDataset(validation_data)
    imdb_test = IMDBDataset(test_data)

    # construct data loader
    train_loader = torch.utils.data.DataLoader(dataset=imdb_train,
                                               batch_size=batch_size,
                                               collate_fn=imdb_collate_func,
                                               shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=imdb_validation,
                                                    batch_size=batch_size,
                                                    collate_fn=imdb_collate_func,
                                                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=imdb_test,
                                              batch_size=batch_size,
                                              collate_fn=imdb_collate_func,
                                              shuffle=False)

    # setup a model
    model = FastText(vocab_size, emb_dim)
    criterion = nn.BCELoss()
    if opt_method == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if lr_annealing:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_annealing)
    train_acc_history = []
    validation_acc_history = []
    train_loss = []
    stop_training = False

    # train
    for epoch in range(num_epochs):
        if lr_annealing:
            lr_scheduler.step()
        for i, (data, lengths, labels) in enumerate(train_loader):
            data_batch, length_batch, label_batch = Variable(data), Variable(lengths), Variable(labels)
            optimizer.zero_grad()
            outputs = model(data_batch, length_batch)
            loss = criterion(outputs, label_batch.float())
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())  # loss.data[0] for lower versions of pytorch
            train_acc = compute_accu(train_loader, model)
            train_acc_history.append(train_acc)
            val_acc = compute_accu(validation_loader, model)
            validation_acc_history.append(val_acc)

            if verbose:  # report performance
                if (i+1) % (batch_size*4) == 0:
                    train_acc = compute_accu(train_loader, model)
                    print('Epoch: [{0}/{1}], Step: [{2}], Loss: {3}, Train Acc: {4}%, Validation Acc:{5}%'.format(
                        epoch+1, num_epochs, i+1, loss.item(), train_acc, val_acc))

            # check if we need to early stop the model
            stop_training = earily_stop(validation_acc_history)
            if stop_training:
                print("early stop triggered")
                break
        # because of the the nested loop
        if stop_training:
            break
    if save_model:
        torch.save(model, './trained/model_{}.pkl'.format(spec))

    if test_model:
        logger.info("Test performance: Test Accu:{}%.".format(compute_accu(test_loader, model)))

    return model, train_loss, train_acc_history, validation_acc_history

