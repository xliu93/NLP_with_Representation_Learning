import spacy
import string
from collections import Counter
from sklearn.feature_extraction import stop_words

PAD_TOKEN = '<pad>'
PAD_IDX = 0
UNK_TOKEN = '<unk>'
UNK_IDX = 1


def extract_ngram_from_text(text, ngram_n, use_spacy=True,
                            remove_stopwords=True, remove_punc=True):
    """
    Function that retrieves all n-grams from the input string
    @param text: raw string
    @param ngram_n: integer that tells the model to retrieve all k-gram where k<=n
    @param remove_stopwords: True or false
    @return ngram_counter: a counter that maps n-gram to its frequency
    @return tokens: a list of parsed ngrams
    """
    # tokenize words
    tokens = tokenization(text, use_spacy, remove_stopwords, remove_punc)
    # extract n grams
    all_ngrams = []
    if ngram_n <= 0:
        return None
    ngram_counter = Counter(tokens)
    all_ngrams += list(ngram_counter.keys()) # unigram is recorded
    if ngram_n > 1:
        for k in range(2, ngram_n + 1):
            kgrams = [tuple(tokens[i:i+k]) for i in range(len(tokens)+k-1)]
            counter_k = Counter(kgrams)
            all_ngrams += list(counter_k.keys())
            ngram_counter += counter_k
    return ngram_counter, all_ngrams


def tokenization(text, use_spacy=True, remove_stopwords=True, remove_punc=True):
    if use_spacy:
        tokenizer = spacy.load('en_core_web_sm')
        tokens = tokenizer(text)
        tokens = [t.text.lower() for t in tokens]
    else:
        tokens = text.lower().split(" ")
    if remove_stopwords:
        tokens = list(filter(lambda t: t not in stop_words.ENGLISH_STOP_WORDS, tokens))
    if remove_punc:
        tokens = list(filter(lambda t: t not in string.punctuation, tokens))
    return tokens


def construct_ngram_indexer(ngram_counter_list, topk):
    """
    Function that selects the most common topk ngrams
    @param ngram_counter_list: list of counters
    @param topk: (int) vocabulary size
    @return ngram2idx: a dictionary that maps ngram to an unique index
    """

    # find the top-k ngram
    all_ngrams = Counter()
    for c in ngram_counter_list:
        all_ngrams.update(c)
    topk_ngrams = all_ngrams.most_common(topk)
    # maps the ngram to an unique index
    ngram_indexer = {k[0]: 2+topk_ngrams.index(k) for k in topk_ngrams}
    # save index=1 for PAD symbol
    # save index=0 for UNKnown symbol
    ngram_indexer.update({PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX})
    return ngram_indexer


def token_to_index(tokens, ngram_indexer):
    """
    Function that transform a list of tokens to a list of token index.
    @param tokens: list of ngram
    @param ngram_indexer: a dictionary that maps ngram to an unique index
    """
    # avoid assigning any ngram to index 1 which is reserved for PAD token
    index_list = [ngram_indexer[t] if t in ngram_indexer.keys() else UNK_IDX for t in tokens]
    return index_list


def process_text_dataset(dataset, n, topk=None, ngram_indexer=None,
                         use_spacy=True, remove_stopwords=True, remove_punc=True):
    """
    Top level function that encodes each datum into a list of ngram indices
    """
    # extract n-gram
    for i in range(len(dataset)):
        text_datum = dataset[i].raw_text
        ngrams, tokens = extract_ngram_from_text(text_datum, n, use_spacy, remove_stopwords, remove_punc)
        dataset[i].set_ngram(ngrams)
        dataset[i].set_tokens(tokens)
        if i % 20 == 0:
            print("===== Sample", i, "=====", dataset[i].tokens)
    # select top k ngram
    if ngram_indexer is None:
        ngram_indexer = construct_ngram_indexer([datum.ngram for datum in dataset], topk)
    # vectorize each datum
    for i in range(len(dataset)):
        dataset[i].set_token_idx(token_to_index(dataset[i].tokens, ngram_indexer))
    return dataset, ngram_indexer