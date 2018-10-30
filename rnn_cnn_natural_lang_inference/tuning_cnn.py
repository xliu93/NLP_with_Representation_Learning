"""
Tuning Dimensional Hyperparameters for CNN-based Model
Notice: for tuning training-related hyperparameters only.
Models that depends on different DataLoaders not applicable.
"""
import logging
import logging.config
import pandas as pd

from constants import HParamKey, DATA_PATH
from constants import DefaultConfig as config
from supervisor import Supervisor
import util

# config logger
logger = logging.getLogger('__main__')
util.init_logger()

# tuning sets
hidden_sets = [50, 100, 200, 500]
kernel_sets = [3, 5, 9]
# dropout_sets = [0, 0.1, 0.2, 0.3, 0.5]
# lr_sets = [0.01, 0.005, 0.001]

spv = Supervisor(config)
spv.load_data()
spv.get_dataloader()

record_save_path = util.get_results_folder()  # or base_path=DATA_PATH
tuning_records = []
for hidden in hidden_sets:
    for kernel in kernel_sets:
        # new hyperparams
        conf_update = {HParamKey.HIDDEN_SIZE: hidden,
                       HParamKey.KERNEL_SIZE: kernel}
        # get output filename
        fname = record_save_path + 'cnn'
        for k, v in conf_update.items():
            fname += '_{}{}'.format(k[:4], v)
        # add output_path to config
        conf_update[HParamKey.MODEL_SAVE_PATH] = record_save_path
        conf_update['model_name'] = fname
        # update
        spv.overwrite_config(conf_update)
        # reset model
        spv.init_cnn_model()
        # train
        val_acc, best_acc = spv.train_model()
        conf_update.update({'val_acc': val_acc, 'best_val_acc': best_acc})
        # save
        tuning_records.append(conf_update)
        spv.save_records(filename=fname + '.csv')

pd.DataFrame.from_records(tuning_records).to_csv(record_save_path + 'cnn_dim_hparams.csv')
