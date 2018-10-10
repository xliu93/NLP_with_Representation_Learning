## Bag of N-gram Document Classification
### 1. Introduction
This is a simple code package of a Document Classification task using Bag-of-Ngram. 

### 2. Overview

	```
	bag_of_ngram_document_classification
	 |- constants.py		# some configs
	 |- model.py			# implementation of learning model 
	 |- structures.py		# data structure for pytorch
	 |- ngram_extraction.py	# feature extraction of N-grams
	 |- save_ngrams.py  	# dump pickle files for reusable tokenized data
	 |- train.py			# train a model with single set of parameters
	 |- tuning.py			# hyperparameter tuning
	 |- utils.py			# tool functions
	 |- aclImdb
	 	   |- ... 	 	# downloaded and extracted data of IMDB reviews
	 |- processed
	 	   |- xx.p  	# tokenized train/validation/test data and ngram indexers
	 |- log
	 	   |- xx.log  	# log files 
	 ``` 
 
### 3. Reference
- lab3 session of Bag-of-words
- DS-GA 1011 Fall 2017 Hw1
