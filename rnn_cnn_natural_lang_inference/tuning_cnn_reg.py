"""
Tuning Regularization Hyperparameters for CNN-based Model

"""
import logging
import logging.config
import pandas as pd

from constants import HParamKey, DATA_PATH, EncoderType
# from constants import DefaultConfig as config
from best_config import OptimalRNNConfig as config
from supervisor import Supervisor
import util

# config logger
logger = logging.getLogger('__main__')
util.init_logger()

# tuning sets
dropout_sets = [0, 0.1, 0.2, 0.3]
wdecay_sets = [0, 1e-7, 1e-5, 1e-3, 1e-1]
# 20 pairs

spv = Supervisor(config)
spv.load_trained_emb()
spv.load_data()
spv.get_dataloader()

record_save_path = util.get_results_folder(base_path='/scratch/xl2053/nlp/hw2_data/results_cnn/')  # or base_path=DATA_PATH
logger.info("Output directory (loss history, accuracy history, checkpoints): {}".format(record_save_path))

tuning_records = []
for drop_p in dropout_sets:
    for weight_decay in wdecay_sets:
        # new hyperparams
        conf_update = {HParamKey.ENCODER_TYPE: EncoderType.CNN,
                       HParamKey.DROPOUT_PROB: drop_p,
                       HParamKey.WEIGHT_DECAY: weight_decay}
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


pd.DataFrame.from_records(tuning_records).to_csv(record_save_path + 'cnn_reg_hparams.csv')
logger.info("Output directory (loss history, accuracy history, checkpoints): {}".format(record_save_path))
logger.info("Tuning finished! ")
