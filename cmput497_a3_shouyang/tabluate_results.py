# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 20:55:55 2019

Takes a tagged text file and the evaluation standard of that file and outputs a tsv comparison of the two.

# Argument 1: A3Data Formatted Original File
# Argument 2: Tagged Text File
# Argument 3: The training file used to generate the model.


@author: Daniel Zhou
"""
import sys
import collections


def read_original_file(filepath):
    """ Reads data in the two column format of A3DataCleaned.

        Returns a list of 2-tuples of word:tag pairs and a word counter.
    """
    # Phase 1: Obtain a word count of training dataset
    word_counter = collections.Counter()
    lines = open(filepath, "r").read().split("\n")
    for line in lines:
        if line:
            word = line.split()[0]
            word_counter[word] += 1

    output = []
    lines = open(filepath, "r").read().split("\n")
    for line in lines:
        if line:
            word = line.split()[0]
            tag  = line.split()[-1]

            output.append((word,tag))

    return output, word_counter

def read_tagged_file(filepath):
    output = []
    lines = open(filepath, "r").read().split("\n")
    for line in lines:
        if line:
            line = line.split()

            for term in line:
                word = term.split("_")[0]
                tag  = term.split("_")[-1]

                output.append((word,tag))
    return output


if __name__ == "__main__":

    original_file = sys.argv[1]
    tagged_file   = sys.argv[2]
    train_file    = sys.argv[3]

    original_seq, original_counts = read_original_file(original_file)
    tagged_seq = read_tagged_file(tagged_file)

    train_seq, train_counts   = read_original_file(train_file)


    for original, tagged in zip(original_seq, tagged_seq):

        if tagged[0] in train_counts:
            print(original[0], original[1], tagged[0], tagged[1], train_counts[tagged[0]], sep="\t")
        else:
            print(original[0], original[1], tagged[0], tagged[1], train_counts[tagged[0]], sep="\t")