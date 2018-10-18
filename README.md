## NLP with Representation Learning

Practice jupyter notebooks for RNN, encoder-decoder system, attention, LSTM staff implemented in pyTorch. 

### Overview

#### 1. Tutorials
- 01. PyTorch Basics
- 02. Logistic Regression (sklearn/pytorch)
- 03. Deep Learning Training Workflow
- 04. Building a Bag-of-Words Model
- 05. Document Classification with FastText
- 06. RNN/CNN based Language Classification
- 07. Intrinsic Evaluation of Word Vectors (GloVe/FastText)
- 08. Language Modeling with KenLM

#### 2. Experiments
- Bag-of-Ngram and Document Classification
- RNN/CNN based Natural Language Inference

### Get started
(for MacOS)

1. Download and install conda (Python3.6)
    - XCode is assumed
    - run `bash [installer_file_that_ends_with.sh]`
    - run `conda list` to confirm that installation succeeded
2. Setup conda environment and install jupyter notebook

    ```
    $ conda create -n learn_nlp python=3.6
    $ conda activate learn_nlp
    $ conda install jupyter notebook matplotlib scikit-learn
    $ conda install -c conda-forge jupyterlab  # very helpful
    ```
    
3. Install pyTorch
   
   ```
   conda install pytorch torchvision -c pytorch
   ```

