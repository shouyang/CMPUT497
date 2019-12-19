import nltk
import os
import sys
import re
import math

# Helpful Tokens used in the program
TOKENS = {
  "START" : "<S>",  # Start of Sentence
  "END"   : "</S>", # End of Sentence
  "UNK"   : "<UNK>" # Unknown Word
}

# Choice of N per the different methods
N = {
    "unsmoothed"    : 1,
    "laplace"       : 1,
    "interpolation" : 3
}

# File listing all filepaths used for training models.
TRAINING_FILES = "train_filepaths.txt"

class N_Model:
    """ This class represents the language model.
    """

    def __str__(self):
        return "N_Model({0}, {1}, n = {2})".format(self.file, self.smoothing, self.N)

    def __init__(self, train_filepath, smoothing = "unsmoothed", UNK_threshold = 1):

        self.train_filepath = train_filepath
        self.file           = re.split(r"/|\\", train_filepath)[-1]
        self.N              = N[smoothing]
        self.smoothing      = smoothing
        self.UNK_threshold  = UNK_threshold

        self.text           = open(train_filepath, "r").read()
        # self.text           = self.gen_adjusted_text(self.UNK_threshold)
        self.freq_dist      = self.get_ngram_freq(self.N)
        self.prev_dist      = self.get_ngram_freq(self.N - 1)
        self.uni_dist       = self.get_ngram_freq(1)

        self.uni_dist[("<UNK>",)] = 1

        if self.smoothing == "interpolation":
            self.interp_coeff = self.gen_interpolation_coeff()

    def gen_interpolation_coeff(self):
        """ Generates linear interpolation method's coefficents for a trigram model.
            Uses deleted interpolation to do so.
        """
        L1 = 0
        L2 = 0
        L3 = 0

        for ngram in self.freq_dist:

            trigram = ngram
            bigram  = ngram[1:]
            unigram = (ngram[-1],)

            nom_trigram = self.freq_dist[trigram] - 1
            nom_bigram  = self.prev_dist[bigram]  - 1
            nom_unigram = self.uni_dist[unigram]  - 1

            denom_trigram = self.prev_dist[trigram[:-1]] - 1
            denom_bigram  = self.uni_dist[(bigram[0],)]  - 1
            denom_unigram = sum(self.uni_dist.values()) - 1

            p_L3 = 0
            if denom_trigram:
                p_L3 = nom_trigram / denom_trigram

            p_L2 = 0
            if denom_bigram:
                p_L2 = nom_bigram / denom_bigram

            p_L1 = 0
            if denom_unigram:
                p_L1 = nom_unigram / denom_unigram

            p_max = max(p_L3, p_L2, p_L1)
            if p_L3 == p_max:
                    L3 += nom_trigram + 1
            if p_L2 == p_max:
                    L2 += nom_bigram  + 1
            if p_L1 == p_max:
                    L1 += nom_unigram + 1

        total_L = sum([L1,L2,L3])
        return [L1 / total_L, L2 / total_L, L3 / total_L]

    def gen_adjusted_text(self, UNK_threshold):
        """ Used to preprocess text, substitutes word occurances below the threshold to the
            unknown token.
        """
        temp_text = self.text[:]

        word_count = self.get_ngram_freq(1)
        for word_tuple, word_count in word_count.items():
            if word_count == 1:
                temp_text = re.sub(r"\b{0}\b".format(word_tuple[0]), " " + TOKENS["UNK"] + " ", temp_text)

        return temp_text


    def get_ngrams(self, n = None):
        """ Generates list of ngrams used to generate ngram distribution.
            Preprocessing step used in initalization.
        """
        if not n:
            n = self.N

        temp = []
        for sent in self.get_sentences():
            temp.extend(sent.split())

        return nltk.ngrams(temp, n)

    def get_ngram_freq(self, n = None):
        """ Generates ngram freqency distributions after preprocessing raw text.
        """
        if not n:
            n = self.N

        dist = nltk.FreqDist(self.get_ngrams(n))
        return dist

    def get_sentences(self):
        """ Generates adjusted sentences from raw input text.
            Appends a sentence start and sentence end token to each sentence.
            Preprocessing step used in initalization.
        """
        return [TOKENS["START"] + " " + s + " " + TOKENS["END"] for s in self.text.split('\n')]

    def eval_text(self, text):
        """ Evaluates a string as per the model trained on the training text.
        """
        # Pre-process sentence given
        sents = text.split('\n')
        words = []
        for sent in sents:
            words.extend(sent.split())

        for idx, word in enumerate(words):
            if (word, ) not in self.uni_dist:
                words[idx] = TOKENS["UNK"]

        # Compute Log-Probablities
        log_prob = 0
        for ngram in nltk.ngrams(words, self.N):
            log_prob += self.eval_ngram(ngram)

        # Compute Perplexity
        num_words =  len(words) - words.count(TOKENS["START"])
        perplexity = 2 ** ((-1 / num_words) * log_prob)

        return perplexity


    def eval_ngram(self, ngram):
        """ Scores the ngram based on the model. Returns a log-prob.
        """
        # Get raw value-
        nom   = self.freq_dist[ngram]
        if self.N == 1:
            nom   = self.uni_dist[ngram]
            denom = sum(self.uni_dist.values())
        else:
            denom = self.prev_dist[ngram[:-1]]

        if self.smoothing == "laplace":
            nom   += 1
            denom += len(self.freq_dist)

        if self.smoothing == "interpolation":
            nom_trigram = self.freq_dist[ngram]
            denom_trigram = self.prev_dist[ngram[1:]]

            # Calculate Trigram prob.
            if nom_trigram == 0 or denom_trigram == 0:
                p_trigram = 0
            else:
                p_trigram = nom_trigram /denom_trigram

            # Calculate Bigram prob.
            nom_bigram   = self.prev_dist[ngram[1:]]
            denom_bigram = self.uni_dist[tuple(ngram[1:2])]

            if nom_bigram == 0 or denom_bigram == 0:
                p_bigram = 0
            else:
                p_bigram = nom_bigram /denom_bigram

            # Calculate Unigram prob.
            nom_unigram   = self.uni_dist[(ngram[-1],)]
            denom_unigram = sum(self.uni_dist.values())

            if nom_unigram == 0 or denom_unigram == 0:
                p_unigram = 0
            else:
                p_unigram = nom_unigram /denom_unigram

            # Return combined prob.
            adj_p_unigram = p_unigram * self.interp_coeff[0]
            adj_p_bigram  = p_bigram  * self.interp_coeff[1]
            adj_p_trigram = p_trigram * self.interp_coeff[2]

            return math.log2(adj_p_unigram + adj_p_bigram + adj_p_trigram)

        return math.log2(nom/denom)


def generate_evaluations_file(train_filepaths, eval_filepaths, output_filepaths, smoothing):
    output = []

    models = [N_Model(fp, smoothing) for fp in train_filepaths]

    for eval_filepath in eval_filepaths:

        eval_text = open(eval_filepath, "r").read()
        eval_file = re.split(r"/|\\", eval_filepath)[-1]

        perplexity_scores = []
        for model in models:
            perplexity_scores.append(model.eval_text(eval_text))

        min_score  = min(perplexity_scores)
        min_index  = perplexity_scores.index(min(perplexity_scores))
        min_model  = models[min_index]

        min_score  = round(min_score, 2)

        output_tuple = tuple([eval_file, min_model.file, min_score, min_model.N])

        output.append(output_tuple)

    output.sort(key = lambda x: x[0])
    with open(output_filepaths, "w") as output_file:
        for item in output:
            output_file.write("{0}\t{1}\t{2}\t{3}\n".format(*item))

def print_usage():
    print("langid: langid -mode -inputfile -outputfile")
    print("\t -mode: One of --unsmoothed --laplace --interpolation")
    print("\t -inputfile: File of filepaths to parse, newline delimited")
    print("\t -outputfile: Name of desired output tsv file ")



def main():
    if len(sys.argv) != 4:
        print_usage()
        return 0

    SMOOTHING      = sys.argv[1].replace("--", "")
    EVAL_FILEPATHS = open(sys.argv[2]).read().split('\n')
    OUTPUT_FILE_PATH = sys.argv[3]

    train_filepaths =   open(TRAINING_FILES, "r").read().split('\n')

    generate_evaluations_file(train_filepaths, EVAL_FILEPATHS, OUTPUT_FILE_PATH, SMOOTHING)


if __name__ == "__main__":
    main()










