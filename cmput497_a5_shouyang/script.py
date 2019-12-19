import nltk
import numpy as np
import csv
import re
import math
import statistics

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.probability import FreqDist 
from nltk.corpus import wordnet


class NBModel:
    """ My implementation of the naive bayes classfier, to be trained on a binary feature dictionary.
    """

    def __init__(self, train_samples, smoothing = 0.5):
        """ Constructs model via training on input training data of samples objects.
        """

        # My model just uses this as a convenient counter data structure for terms.
        self.label_freq        = FreqDist()
        self.term_label_freq   = FreqDist()
        self.term_label_counts = FreqDist()
        
        self.smoothing = smoothing

        self.train(train_samples)

    def train(self, train_samples):
        """ Updates the model internal freqency distributions from a list of samples.
        """
        for sample in train_samples:
        
            sample_features = sample[0]
            sample_label    = sample[1]

            self.label_freq[sample_label] += 1

            for feature, bl in sample_features.items():
                if bl == True:
                    label_feature = (sample_label, feature)
                    self.term_label_freq[label_feature]  += 1
                    self.term_label_counts[sample_label] += 1

    def classify(self, features):
        """ Classifies a binary feature dictionary 
        """

        label_probs = {}

        # Generate Inital P(C)
        label_smoothing_term   = self.smoothing / (self.smoothing * self.label_freq.N())
        feature_smoothing_term = self.smoothing / (self.smoothing * self.term_label_freq.N())

        for label in self.label_freq:
            label_probs[label] = math.log2((self.label_freq[label] / self.label_freq.N()) + label_smoothing_term)

            denom = self.term_label_counts[label]            
            # Generate Per Term P(D|C)     
            for feature, bl in features.items():

                if bl:
                    key = (label, feature)
                    nom  = self.term_label_freq[key]
                    label_probs[label] += math.log2((nom  / denom) + feature_smoothing_term)

        # Return Max Label
        return max(label_probs, key = label_probs.get)

class Sample:
    """ Defines a sample from one row of the given assignment datasets. 
    """
    def __init__(self, label, text):
        self.raw_text = text
        self.label = label
        self.text = self.gen_text(text)
        self.features = None

    def __str__(self):
        return self.text[:125] + "..."

    def gen_text(self, text):
        """ Preprocesses the raw text from the sample.
        """
        replacement_regex = "[0-9]+"
        return re.sub(replacement_regex, "D" , self.raw_text)

    def gen_features(self, word_features):
        """ Updates this sample's features according to some predefined feature universe as defined as a list "word_features".
        """
        output = {}


        lemmatizer = WordNetLemmatizer() 
        sample_terms = set(lemmatizer.lemmatize(word, "v") for sent in sent_tokenize(self.text) for word in word_tokenize(sent))

        for word in word_features:
            output[word] = (word in sample_terms)

        self.features = output

def read_data(filepath):
    """
    Reads comma separated data from a filepath, assumes the files contain a header and are in assigment format (ie. 2-tuple of "type" and "text").

    Creates a list of sample objects.
    """
    output = []

    with open(filepath, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter = ",")
        for idx, row in enumerate(reader, 1):
            assert(len(row) == 2) # Make sure that csv file doe snot use the delimiter in text
            if idx == 1: # Skip header row - assuming to be the first in each file.
                continue

            sample = Sample(label = row[0], text = row[1])
            output.append(sample)

    return output

def generate_freq_dist(samples):
    fdist = FreqDist()
    lemmatizer = WordNetLemmatizer() 

    for sample in samples:
        temp = FreqDist([lemmatizer.lemmatize(word, "v") for sent in sent_tokenize(sample.text) for word in word_tokenize(sent)])
        fdist.update(temp) 
    return fdist

def do_cv(all_samples, N = 3):
    accuracy = []
    for i in range(N): # For each fold
        sample_sets = [list(chunk) for chunk in np.array_split(all_samples,N)]

        # Generate Test and Train Set
        test_set   = sample_sets.pop(i)
        train_set =  []
        for chunk in sample_sets:
            train_set.extend(chunk)

        # Generate Feature Vector
        train_freqdist = generate_freq_dist(train_set)
        train_features = [x[0] for x in train_freqdist.most_common(MAX_FEATURES)]        
        train_features = train_features[SKIP_N_FEATURES:]
        
        # Generate Feature Vector Per Sample
        for test_sample in test_set:
            test_sample.gen_features(train_features)

        for train_sample in train_set:
            train_sample.gen_features(train_features)

        # Train Model
        train_data = [(sample.features, sample.label) for sample in train_set]
        classifier = NBModel(train_data)

        # Generate Accuracy on Test
        correct_items = 0
        for test_sample in test_set:
            est    = classifier.classify(test_sample.features)
            actual = test_sample.label

            if est == actual:
                correct_items += 1

        fold_acc = correct_items / len(test_set)
        
        accuracy.append(fold_acc)
        print("K-FOLD CV {0}: ACCURACY = {1}".format(i, fold_acc))
        
        # Print Samples
        for sample in test_set:
            print("K-FOLD CV {0}: TEST SAMPLE  = {1}".format(i, sample))

        for sample in train_set:
            print("K-FOLD CV {0}: TRAIN SAMPLE  = {1}".format(i, sample))
    
    print("*** Accuracies per Fold: {0}".format(accuracy))
    print("*** Mean Accuracy Per Fold: {0}".format(statistics.mean(accuracy)))

def do_test(train_samples, test_samples, testfile_name):
    # Generate Feature Vector
    train_freqdist = generate_freq_dist(train_samples)
    train_features = [x[0] for x in train_freqdist.most_common(MAX_FEATURES)]        
    train_features = train_features[SKIP_N_FEATURES:]

    # Generate Features per Sample
    for test_sample in test_samples:
        test_sample.gen_features(train_features)

    for train_sample in train_samples:
        train_sample.gen_features(train_features)

    # Train Model
    train_data = [(sample.features, sample.label) for sample in train_samples]
    classifier = NBModel(train_data)

    # Evaluate On Test Data
    correct_items = 0
    for test_sample in test_samples:
        est    = classifier.classify(test_sample.features)
        actual = test_sample.label

        if est == actual:
            correct_items += 1

    accuracy = correct_items / len(test_samples)
    print("*** Accuracy On Test Data {1}: {0}".format(accuracy, testfile_name))

    # Generate CSV File
    filename = "output_{}.csv".format(testfile_name.split(".")[0])

    with open(filename, "w") as file:
        file.write("original_label, classifier_label, text\n")
        for sample in test_samples:
            file.write("{},{},{}\n".format(sample.label,classifier.classify(sample.features), sample.raw_text))


if __name__ == "__main__":

    TRAIN_DATA = "trainBBC.csv"
    TEST_DATA  = "testBBC.csv"
    EVAL_DATA  = "evalBBC.csv"

    SKIP_N_FEATURES = 100 
    MAX_FEATURES    = 600

    train_samples = read_data(TRAIN_DATA)
    test_samples  = read_data(TEST_DATA)
    eval_samples  = read_data(EVAL_DATA)

    do_cv(train_samples, 3)
    do_test(train_samples, test_samples, TEST_DATA)
    do_test(train_samples, eval_samples, EVAL_DATA) # Shortcut for generating the necessary reporting elements of the eval file.