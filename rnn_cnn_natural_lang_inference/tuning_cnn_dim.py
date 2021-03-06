"""
Tuning Dimensional Hyperparameters for CNN-based Model
Notice: for tuning training-related hyperparameters only.
Models that depends on different DataLoaders not applicable.
"""
import logging
import logging.config
import pandas as pd

from constants import HParamKey, DATA_PATH, EncoderType
from constants import DefaultConfig as config
from supervisor import Supervisor
import util

# config logger
logger = logging.getLogger('__main__')
util.init_logger()

# tuning sets
hidden_sets = [50, 100, 200, 500]
kernel_sets = [3, 5, 9, 15]
lr_sets = [0.01, 0.005, 0.001]

# get supervisor
spv = Supervisor(config)
spv.load_data()
spv.get_dataloader()

record_save_path = util.get_results_folder()  # or base_path=DATA_PATH
tuning_records = []
for hidden in hidden_sets:
    for kernel in kernel_sets:
        for lr in lr_sets:
            # new hyperparams
            conf_update = {HParamKey.ENCODER_TYPE: EncoderType.CNN,
                           HParamKey.HIDDEN_SIZE: hidden,
                           HParamKey.KERNEL_SIZE: kernel,
                           HParamKey.LEARNING_RATE: lr}
            # get output filename
            model_name = ''
            for k, v in conf_update.items():
                model_name += '{}{}'.format(k[:3], v)
            # add output_path to config
            conf_update[HParamKey.MODEL_SAVE_PATH] = record_save_path
            conf_update['model_name'] = model_name
            # update
            spv.overwrite_config(conf_update)
            # reset model
            spv.init_model()
            # train
            val_acc, best_acc = spv.train_model()
            conf_update.update({'val_acc': val_acc, 'best_val_acc': best_acc})
            # record
            tuning_records.append(conf_update)
            # save
            spv.save_records(filename=record_save_path + model_name + '.csv')

pd.DataFrame.from_records(tuning_records).to_csv(record_save_path + 'cnn_dim_hparams.csv')
logger.info("Output directory (loss history, accuracy history, checkpoints): {}".format(record_save_path))
logger.info("Tuning finished! ")
