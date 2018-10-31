import os
import logging
import logging.config
import pandas as pd
import gc
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F

from constants import (HParamKey, TrainRecordKey as trKey, CheckpointKey as cpKey,
                       EncoderType, LoaderType as lType, DEVICE)
from util import get_fasttext_embedding
from data_loader import get_snli_data, snli_token2id, snliDataset, snli_collate_func
from models.RNN import RNNModel
from models.CNN import CNNModel

logger = logging.getLogger('__main__')


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.word2idx = None
        self.idx2word = None
        self.trained_emb = None
        self.datasets = {}
        self.loaders = {}
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.records = {}

    def overwrite_config(self, config_new):
        self.config.update(config_new)
        self.records = {}  # clear records
        self.model = None  # reset model
        self.optimizer = None
        self.lr_scheduler = None
        gc.collect()

    def load_data(self):
        corpus_name = self.config[HParamKey.FT_CORPUS_NAME]
        vocab_size = self.config[HParamKey.VOCAB_SIZE]

        # Load pre-trained embeddings of FastText
        self.word2idx, self.idx2word, self.trained_emb = get_fasttext_embedding(vocab_size, corpus_name)
        logger.info("Pre-trained embeddings loaded!")

        # Load train/validation sets
        train_set, val_set = get_snli_data()
        logger.info("\n===== train/validation sets =====\nTrain sample: {}\nValidation sample: {}".format(
            len(train_set), len(val_set)
        ))
        self.datasets[lType.TRAIN] = train_set
        self.datasets[lType.VAL] = val_set

    def get_dataloader(self):
        batch_size = self.config[HParamKey.BATCH_SIZE]

        # Convert to indices based on FastText vocabulary
        train_prem, train_hypo, train_label = snli_token2id(self.datasets[lType.TRAIN], self.word2idx)
        val_prem, val_hypo, val_label = snli_token2id(self.datasets[lType.VAL], self.word2idx)
        logger.info("Converted to indices! ")

        # Create DataLoader
        logger.info("Creating DataLoader...")
        train_dataset = snliDataset(train_prem, train_hypo, train_label)
        self.loaders[lType.TRAIN] = torch.utils.data.DataLoader(dataset=train_dataset,
                                                          batch_size=batch_size,
                                                          collate_fn=snli_collate_func,
                                                          shuffle=True)
        val_dataset = snliDataset(val_prem, val_hypo, val_label)
        self.loaders[lType.VAL] = torch.utils.data.DataLoader(dataset=val_dataset,
                                                        batch_size=batch_size,
                                                        collate_fn=snli_collate_func,
                                                        shuffle=True)
        logger.info("DataLoader generated!")

    def init_model(self):
        """
        Initialize a classification model with selected encoder type.
        """
        enc_type = self.config[HParamKey.ENCODER_TYPE]
        if enc_type == EncoderType.RNN:
            self.init_rnn_model()
        elif enc_type == EncoderType.CNN:
            self.init_cnn_model()
        else:
            logger.error("Unknown encoder type! Available Encoders: [RNN, CNN]")
            exit(1)

    def init_rnn_model(self):
        """Get an instance of parametrized RNN model"""
        self.model = RNNModel(
            hidden_size=self.config[HParamKey.HIDDEN_SIZE],
            num_classes=self.config[HParamKey.NUM_CLASS],
            num_layers=self.config[HParamKey.NUM_LAYER],
            dropout_p=self.config[HParamKey.DROPOUT_PROB],
            trained_emb=self.trained_emb
        ).to(DEVICE)  # make sure the model moved to device
        logger.info("Initialized a RNN model:\n{}".format(self.model))

    def init_cnn_model(self):
        """Get an instance of parametrized CNN model"""
        self.model = CNNModel(
            hidden_size=self.config[HParamKey.HIDDEN_SIZE],
            num_classes=self.config[HParamKey.NUM_CLASS],
            num_layers=self.config[HParamKey.NUM_LAYER],
            kernel_size=self.config[HParamKey.KERNEL_SIZE],
            dropout_p=self.config[HParamKey.DROPOUT_PROB],
            trained_emb=self.trained_emb
        ).to(DEVICE)
        logger.info("Initialized a CNN model:\n{}".format(self.model))

    def init_optimizer(self):
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.config[HParamKey.LEARNING_RATE])

    def init_lr_scheduler(self):
        if self.config[HParamKey.LR_DECAY]:
            self.lr_scheduler = opt.lr_scheduler.StepLR(self.optimizer, step_size=1,
                                                        gamma=self.config[HParamKey.LR_DECAY])

    def train_model(self):
        # Initialization for training
        lr_decay = self.config[HParamKey.LR_DECAY]
        num_epochs = self.config[HParamKey.NUM_EPOCHS]
        do_earlystop = self.config[HParamKey.IF_EARLY_STOP]

        # criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        self.init_optimizer()
        self.init_lr_scheduler()

        self.records[trKey.TRAIN_ACC] = []
        self.records[trKey.VAL_ACC] = []
        self.records[trKey.TRAIN_LOSS] = []
        self.records[trKey.VAL_LOSS] = []
        best_val_acc = 0

        # Train loop
        logger.info("Start training...")
        for epoch in range(num_epochs):
            if lr_decay:
                self.lr_scheduler.step()
            for i, (prem, hypo, p_len, h_len, label) in enumerate(self.loaders[lType.TRAIN]):
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(prem, hypo, p_len, h_len)  # input for model.forward()
                loss = criterion(outputs, label)
                loss.backward()
                self.optimizer.step()

                if i > 0 and i % 30 == 0:
                    train_acc, train_loss = self.eval_model(self.loaders[lType.TRAIN])
                    val_acc, val_loss = self.eval_model(self.loaders[lType.VAL])
                    logger.info('(epoch){}/{} (step){}/{} (trainLoss){} (trainAcc){} (valLoss){} (valAcc){}'.format(
                        epoch + 1, num_epochs, i + 1, len(self.loaders[lType.TRAIN]),
                        train_loss, train_acc, val_loss, val_acc))
                    self.records[trKey.TRAIN_ACC].append(train_acc)
                    self.records[trKey.TRAIN_LOSS].append(train_loss)
                    self.records[trKey.VAL_ACC].append(val_acc)
                    self.records[trKey.VAL_LOSS].append(val_loss)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        # todo: save current best if meet the required 65% val accuracy
                        if val_acc > 65:
                            logger.info("Saving current optimal model...")
                            self.save_checkpoint(epoch_idx=epoch)
                    if do_earlystop and self.check_early_stop():
                        logger.info("###### Early stop triggered! ######")
                        break
                if do_earlystop and self.check_early_stop():  # for nested loop
                    break
        # final eval
        val_acc, val_loss = self.eval_model(self.loaders[lType.VAL])
        return val_acc, best_val_acc

    def eval_model(self, loader):
        if self.model is None:
            raise AssertionError("Attempt to evaluate a model not initialized!")
        else:
            correct = 0
            total = 0
            cur_loss = 0
            self.model.eval()
            for prem, hypo, p_len, h_len, labels in loader:
                outputs = F.softmax(self.model(prem, hypo, p_len, h_len), dim=1)
                predicted = outputs.max(1, keepdim=True)[1]
                # compute CEloss
                cur_loss += F.cross_entropy(outputs, labels).cpu().detach().numpy()
                # compute accuracy
                total += labels.size(0)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()
            return 100 * correct / total, cur_loss

    def save_checkpoint(self, epoch_idx):
        filename = self.config[HParamKey.MODEL_SAVE_PATH] + 'checkpoints/' +\
                   self.config['model_name'] + '.tar'
        content = {
            cpKey.MODEL: self.model.state_dict(),
            cpKey.CONFIG: self.config,
            cpKey.OPTIM: self.optimizer.state_dict(),
            cpKey.LR_SCHD: self.lr_scheduler if self.config[HParamKey.LR_DECAY] else None,
            cpKey.EPOCH_IDX: epoch_idx,  # might remove later
            cpKey.RECORDS: self.records  # might remove later
        }
        torch.save(content, filename)

    def load_checkpoint(self, f_path):
        # local test
        if not torch.cuda.is_available():
            content = torch.load(f_path, map_location='cpu')
            # update config
            self.overwrite_config(content[cpKey.CONFIG])
            # update model, optimizer, lr_schedule (if applicable)
            self.init_model()
            self.model.load_state_dict(content[cpKey.MODEL])
            # self.init_optimizer()
            # self.optimizer.load_state_dict(content[cpKey.OPTIM])
            # self.init_lr_scheduler()
            # if self.config[HParamKey.LR_DECAY]:
            #     self.lr_scheduler.load_state_dict(content[cpKey.LR_SCHD])
            return

        content = torch.load(f_path)
        # update config
        self.overwrite_config(content[cpKey.CONFIG])
        # update model, optimizer, lr_schedule (if applicable)
        self.init_model()
        self.model.load_state_dict(content[cpKey.MODEL])
        self.init_optimizer()
        self.optimizer.load_state_dict(content[cpKey.OPTIM])
        self.init_lr_scheduler()
        if self.config[HParamKey.LR_DECAY]:
            self.lr_scheduler.load_state_dict(content[cpKey.LR_SCHD])

    def save_records(self, filename):
        pd.DataFrame(self.records).to_csv(filename)

    def check_early_stop(self):
        val_acc_history = self.records[trKey.VAL_ACC]
        lb = self.config[HParamKey.ES_LOOKBACK]
        if (len(val_acc_history) >= 2 * lb + 1) and sum(val_acc_history[-2*lb:-lb]) > sum(val_acc_history[-lb:]):
            return True
        return False
