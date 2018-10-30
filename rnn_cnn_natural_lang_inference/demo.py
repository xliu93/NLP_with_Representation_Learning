"""
Tuning Dimensional Hyperparameters for RNN-based Model
Notice: for tuning training-related hyperparameters only.
Models that depends on different DataLoaders not applicable.
"""
import time
import logging
import logging.config

from constants import HParamKey
from constants import DefaultConfig as config
from supervisor import Supervisor
import util

# config logger
logger = logging.getLogger('__main__')
util.init_logger(logfile='demo-{}.log'.format(int(time.time())))

# init supervisor
spv = Supervisor(config)
# get DataLoader for supervisor
spv.load_data()
spv.get_dataloader()

# new parameters
conf_update = {HParamKey.HIDDEN_SIZE: 200,
               HParamKey.DROPOUT_PROB: 0.2,
               HParamKey.IF_EARLY_STOP: True}
result_folder = util.get_results_folder()
logger.info("Output directory (loss history, accuracy history, checkpoints): {}".format(result_folder))

# ============ RNN demo ===========
# get output filename
fname = 'demo_rnn'
for k, v in conf_update.items():
    fname += '_{}{}'.format(k[:4], v)
# update supervisor config
config[HParamKey.MODEL_SAVE_PATH] = result_folder
config['model_name'] = fname
spv.overwrite_config(conf_update)
# init model
spv.init_rnn_model()
# train
val_acc, best_acc = spv.train_model()
spv.save_records(filename=fname + '.csv')

# ============ CNN demo ===========
# get output filename
fname = 'demo_cnn'
for k, v in conf_update.items():
    fname += '_{}{}'.format(k[:4], v)
# update supervisor config
config[HParamKey.MODEL_SAVE_PATH] = result_folder
config['model_name'] = fname
spv.overwrite_config(conf_update)
# init model
spv.init_cnn_model()
# train
val_acc, best_acc = spv.train_model()
spv.save_records(filename=fname + '.csv')

