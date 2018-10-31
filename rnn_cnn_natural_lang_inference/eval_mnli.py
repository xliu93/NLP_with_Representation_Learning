"""
Evaluation Trained Model on MultiNLI.
"""
import sys
import logging
import logging.config
import pandas as pd

from constants import (HParamKey, DATA_PATH, TaskName, MultiGenre,
                       DefaultConfig as config, LoaderType)
from supervisor import Supervisor
import util


# parse user input (path of model checkpoint)
if len(sys.argv) < 2:
    print("=== ! ===\nNo input for trained model path!")
    model_save_path = 'results_1540947201/checkpoints/encRNNhid50lea0.01.tar'
    print("Using model ({}) for illustration...".format(model_save_path))
else:
    model_save_path = sys.argv[1]

# config logger
logger = logging.getLogger('__main__')
util.init_logger()
logger.info("Evaluation of trained model on MultiNLI. Model ({})".format(model_save_path))

# init supervisor
spv = Supervisor(config)
# load trained embedding
spv.load_trained_emb()
# load trained model
spv.load_checkpoint(f_path=model_save_path)

eval_results = []
# Evaluate on each genre of MultiSNLI
for genre in MultiGenre:
    logger.info("##### GENRE: {} #####".format(genre.upper()))
    spv.load_data(TaskName.MNLI, genre)
    spv.get_dataloader()
    acc, _ = spv.eval_model(spv.loaders[LoaderType.VAL])
    eval_results.append({'Genre': genre, "valAcc": acc})
    # logger.info("(model){} (genre){} (val_acc){}".format(spv.config['model_name'], genre, acc))

print("\n", pd.DataFrame.from_records(eval_results), "\n")

logger.info("Model from {}\nEvaluation on MultiNLI is done!\n".format(model_save_path))

