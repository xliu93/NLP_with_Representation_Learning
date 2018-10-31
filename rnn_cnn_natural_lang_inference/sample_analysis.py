"""
Sample Analysis on SNLI validation set.
"""
import sys
import logging
import logging.config
import torch.nn.functional as F

from constants import DefaultConfig as config, LoaderType
from data_loader import LABEL_IDX
from supervisor import Supervisor
import util

if len(sys.argv) < 2:
    print("=== ! ===\nNo input for trained model path!")
    model_save_path = 'results_1540947201/checkpoints/encRNNhid50lea0.01.tar'
    print("Using model ({}) for illustration...".format(model_save_path))
else:
    model_save_path = sys.argv[1]

# init supervisor
spv = Supervisor(config)
spv.load_trained_emb()
spv.load_data()
spv.get_dataloader()

# ============= Load checkpoint ============
# load trained model
spv.load_checkpoint(f_path='./results_1540933016/checkpoints/demo_rnn_hidd200_drop0.2_if_eTrue.tar')
acc, loss = spv.eval_model(spv.loaders[LoaderType.VAL])
logger.info("Load model evaluated on val_loader: (valAcc){} (valLoss){}".format(acc, loss))

# ============= Sample Analysis ============
# Find at least 3 samples from correct and
# incorrect classification respectively
spv.model.eval()
corr_count = 0
incorr_count = 0
for prem, hypo, p_len, h_len, labels in spv.loaders[LoaderType.VAL]:
    outputs = F.softmax(spv.model(prem, hypo, p_len, h_len), dim=1)
    predicted = outputs.max(1, keepdim=True)[1]
    eq = predicted.eq(labels.view_as(predicted)).numpy()
    for i in range(len(eq)):
        if eq[i] == 1:  # correct
            corr_count += 1
            print("\nCORRECT:")
        else:
            incorr_count += 1
            print("\nWRONG:")
        print("Premise:", util.back_to_sentence(spv.idx2word, prem.numpy()[i]))
        print("Hypothesis:", util.back_to_sentence(spv.idx2word, hypo.numpy()[i]))
        print("Truth:", LABEL_IDX[labels.numpy()[i]])
        print("Predicted:", LABEL_IDX[predicted.numpy()[i][0]])
        if corr_count > 3 and incorr_count > 6:
            break
    if corr_count > 3 and incorr_count > 6:
        break

print("### Sample Analysis done!###\n")
