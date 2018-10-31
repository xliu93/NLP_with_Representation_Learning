## RNN/CNN-based Natural Language Inference
### 1. Introduction
An implementation of RNN/CNN based models to tackle Stanford Natural Language 
Inference (SNLI) task. And I will further evaluate the trained models on 
Multi-Genre Natural Language Inference (MultiNLI) task.  
The purpose of this little project is to experience what happens when a model 
trained and tuned for one domain or data distribution is used for another domain.

### 2. Overview
```
rnn_cnn_natural_lang_inference
 |- README.md           # Introduction :)
 |- constants.py        # constants, keywords, configs
 |- best_config.py      # partial optimal hyperparameters (for further tuning) 
 |- data_loader.py      # convert raw text to pytorch DataLoader
 |- supervisor.py       # MAIN DRIVER, manage the data loader, model initialization, train and evaluation, etc. 
 |- models              # Implementation of classification models
       |- RNN.py        # Model with RNN(GRU) encoder
       |- CNN.py        # Model with CNN encoder
 |- demo.py             # train a RNN/CNN model from default config
 |- tuning.py           # hyperparameter tuning
 |- utils.py            # tool functions
 |
 |- (following are tuning/eval scripts)
 |- tuning_rnn_dim.py   # tuning dimensional hyperparameters for RNN-based model  
 |- tuning_rnn_reg.py   # tuning regularization hyperparameters for RNN-based model  
 |- tuning_cnn_dim.py   # tuning dimensional hyperparameters for CNN-based model  
 |- tuning_cnn_reg.py   # tuning regularization hyperparameters for CNN-based model  
 |- sample_analysis.py  # Report N>=3 examples from correct and incorrect classification
 |- eval_mnli.py        # Evaluate a selected(best) model on all genres of MultiNLI task
 |- tuning_mnli.py      # Fine tuning on MultiNLI task (Bonus Part)
```
 
### 3. Good Learning Staff
There are some posts helped me understand the mechanism or crack some tricky part of the problem:
- a blog about [how to use pre-trained embeddings](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76)

### 4. Reference
- Lab4 session of rnn/cnn based classification models, you can take a look at my notebook [06.rnn_cnn_lang_classification](https://github.com/xliu93/NLP_with_Representation_Learning/blob/master/06.rnn_cnn_lang_classification.ipynb)
- Pytorch example of [snli](https://github.com/pytorch/examples/tree/master/snli)
- Our team project codebase [translation](https://github.com/ds1011teamproject/translation/tree/master)

### 5. Takeaways
1. When I use pre-trained word embeddings, e.g. FastText, 
I found that using a word embedding trained form larger corpus helps. 
My interpretation is that the larger the language base, the more information
it contains, and the embeddings are less likely to be over-fitted on a subset
of the whole language space.

2. The size of training set matters. This is nothing new. The instructor 
warned us the SNLI dataset is pretty large, so I used a small subset (1/10)
of training samples as my train set before tuning hyperparameters. I was
worried because the classification accuracy was really low on the validation
set, and I was even worried that the model was not learning. But to my relief,
after I switched to the whole train set, it achieved a acceptable accuracy 
after just one epoch.


