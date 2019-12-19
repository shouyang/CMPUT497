# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:55:55 2019

Trains and tags a text to stdout via command line arguments.
    - First cmd argument  should specify a  training path in the format of the files in A3DataCleaned
    - Second cmd argument should specify a  test path.

@author: Daniel Zhou
"""

import nltk
import sys
import collections

from nltk.tag import hmm

UNK_TOKEN = "<UNK>"


def read_train_data(filepath, threshold = 0):
    """ Reads data in the two column format of A3DataCleaned.
    
    Returns nested list of word-tag pairs, inner lists denote sentences.
    """

    # Phase 1: Obtain a word count of training dataset
    word_counter = collections.Counter()
    lines = open(filepath, "r").read().split("\n")
    for line in lines:
        if line:
            word = line.split()[0]
            word_counter[word] += 1

    # Phase 2: Generate nested list of lists object, doing replacement as needed
    lines = open(filepath, "r").read().split("\n")
    output = []
    cur    = []
    for line in lines:
        if line:

            word = line.split()[0]
            tag  = line.split()[-1]

            if threshold and word_counter[word] <= threshold:
                word = UNK_TOKEN

            cur.append(tuple([word,tag]))
        else:
            output.append(cur)
            cur = []

    return output, word_counter

def read_test_data(filepath, word_counts, threshold = 0):
    """ Reads pretokenized text, extracted from the A3DataCleaned sets.
        Each line of the filepath should be a sentence.

        Returns a nested list of lists, inner lists denoting sentences.
    """
    output = []
    
    lines = open(filepath, "r").read().split("\n")

    for line in lines:
        output.append(line.split())

    return output


if __name__ == "__main__":
    train_file = sys.argv[1]
    test_file  = sys.argv[2]

    train_data, word_counts = read_train_data(train_file)
    test_data = read_test_data(test_file, word_counts)

    trainer    = hmm.HiddenMarkovModelTrainer()
    model      = trainer.train_supervised(train_data)

    for sent in test_data:
        if sent:
            tagged_sent = model.tag(sent)


            output = []
            for word, tag in tagged_sent:
                output.append(word +"_" + tag)
            print(" ".join(output))