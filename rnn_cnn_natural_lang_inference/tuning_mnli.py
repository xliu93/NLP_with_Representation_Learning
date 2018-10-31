"""
Fine tuning on MultiNLI.
Select the best model trained on SNLI, train it on each genre ofMultiNLI
for a few more epochs, save each fine-tuned model and later evaluate it
across all genres (use eval_mnli.py).
"""
import sys
import logging
import logging.config

from constants import (HParamKey, DATA_PATH, TaskName, MultiGenre,
                       DefaultConfig as config, LoaderType)
from supervisor import Supervisor
import util


# parse user input (path of model checkpoint)
if len(sys.argv) < 2:
    print("=== ! ===\nNo input for trained model path and number of epochs!")
    model_save_path = 'results_1540947201/checkpoints/encRNNhid50lea0.01.tar'
    more_epochs = 5
    print("Train model ({}) for 5 epochs as illustration...".format(model_save_path))
elif len(sys.argv) < 3:
    print("=== ! ===\nNo input for number of epochs!")
    print("Using num_epochs=5 for trial...")
    more_epochs = 5
    model_save_path = sys.argv[1]
else:
    model_save_path = sys.argv[1]
    more_epochs = int(sys.argv[2])

# config logger
logger = logging.getLogger('__main__')
util.init_logger()
logger.info("Fine-tuning trained model on MultiNLI. Using checkpoint: {}".format(model_save_path))

# init supervisor
spv = Supervisor(config)
# load trained embedding
spv.load_trained_emb()

record_save_path = util.get_results_folder(base_path='/scratch/xl2053/nlp/hw2_data/results_tuning_mnli/')
eval_results = []


# ======== Fine tuning =========
config_update = {
    HParamKey.REPORT_INTERVAL: 5,
    HParamKey.MODEL_SAVE_REQ: 0,  # no min requirement
    HParamKey.BATCH_SIZE: 32,     # smaller batch size considering the train set size
    HParamKey.MODEL_SAVE_PATH: record_save_path
}
for genre in MultiGenre:
    # get data of this genre
    spv.load_data(TaskName.MNLI, genre)
    spv.get_dataloader()
    # get trained model
    spv.load_checkpoint(model_save_path)
    # update config
    model_name = spv.config['model_name'] + '_' + genre
    config_update['model_name'] = model_name
    spv.overwrite_config(config_update, reset=False)  # don't reset, or lose loaded model
    # train a few more epochs
    spv.train_more_epochs(more_epochs)
    _, _ = spv.eval_model(spv.loaders[LoaderType.VAL])
    spv.save_records(filename=record_save_path + model_name + '.csv')

logger.info("Model from {}\nEvaluation on MultiNLI is done!\n".format(model_save_path))

