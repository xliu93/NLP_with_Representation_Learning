"""
Optimal config for RNN/CNN based models
This is only part of the optimal parameters, it is used
for further tuning on regularization parameters.
(Not the final best model)
"""

from constants import HParamKey, DefaultConfig as conf

OptimalRNNConfig = dict()
OptimalRNNConfig.update(conf)
OptimalRNNConfig.update({
    HParamKey.NUM_LAYER: 1,
    HParamKey.HIDDEN_SIZE: 200,
    HParamKey.LEARNING_RATE: 0.005,
    HParamKey.DROPOUT_PROB: 0.2,
    HParamKey.WEIGHT_DECAY: 0.0,
})

OptimalCNNConfig = dict()
OptimalCNNConfig.update(conf)
OptimalCNNConfig.update({
    HParamKey.NUM_LAYER: 2,
    HParamKey.HIDDEN_SIZE: 500,
    HParamKey.KERNEL_SIZE: 3,
    HParamKey.DROPOUT_PROB: 0.2,
    HParamKey.WEIGHT_DECAY: 0.0,
    HParamKey.LEARNING_RATE: 0.001,
})
