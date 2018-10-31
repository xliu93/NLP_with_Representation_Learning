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

result_folder = util.get_results_folder()
logger.info("Output directory (loss history, accuracy history, checkpoints): {}".format(result_folder))

# ============ RNN demo ===========
# new parameters
conf_update = {HParamKey.ENCODER_TYPE: 'RNN',
               HParamKey.HIDDEN_SIZE: 200,
               HParamKey.DROPOUT_PROB: 0.2,
               HParamKey.IF_EARLY_STOP: True}
# get output filename
fname = ''
for k, v in conf_update.items():
    fname += '{}{}'.format(k[:4], v)
# update supervisor config
config[HParamKey.MODEL_SAVE_PATH] = result_folder
config['model_name'] = fname
spv.overwrite_config(conf_update)
# init model
spv.init_model()
# train
val_acc, best_acc = spv.train_model()
logger.info("Performance on val_loader: (valAcc){} (bestAcc){}\n\n".format(val_acc, best_acc))
spv.save_records(filename=result_folder + fname + '.csv')

# ============ CNN demo ===========
conf_update = {HParamKey.ENCODER_TYPE: 'CNN',
               HParamKey.HIDDEN_SIZE: 200,
               HParamKey.DROPOUT_PROB: 0.2,
               HParamKey.IF_EARLY_STOP: True}
# get output filename
fname = ''
for k, v in conf_update.items():
    fname += '{}{}'.format(k[:4], v)
# update supervisor config
config[HParamKey.MODEL_SAVE_PATH] = result_folder
config['model_name'] = fname
spv.overwrite_config(conf_update)
# init model
spv.init_model()
# train
val_acc, best_acc = spv.train_model()
logger.info("Performance on val_loader: (valAcc){} (bestAcc){}\n\n".format(val_acc, best_acc))
spv.save_records(filename=result_folder + fname + '.csv')

logger.info("Demo done!")
