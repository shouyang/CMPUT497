import nltk
import os
import sys
import re
import math

# Helpful Tokens used in the program
TOKENS = {
  "START" : "<S>",  # Start of Sentence
  "END"   : "</S>", # End of Sentence
  "UNK"   : "<UNK>" # Unknown Token
}

# Choice of N per the different methods
N = {
    "unsmoothed"    : 1,
    "laplace"       : 2,
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
            self.freqs = [self.get_ngram_freq(n) for n in range(0,self.N + 1)]

    def gen_interpolation_coeff(self):
        """ Generates linear interpolation method's coefficents for a trigram model.
            Uses deleted interpolation to do so.
        """

        L_counts = [0 for x in range(self.N)]
        ngram_freqs = [self.get_ngram_freq(n) for n in range(0,self.N + 1)]

        for ngram in self.freq_dist:

            c_at_n = [0 for x in range(self.N)]
            p_at_n = [0 for x in range(self.N)]
            
            for n_len in range(1, self.N + 1):

                nom_ngram   = ngram[len(ngram) - n_len:]
                denom_ngram = nom_ngram[:-1]

                nom_freq    = ngram_freqs[n_len]
                denom_freq  = ngram_freqs[n_len - 1 ]

                nom_count   = nom_freq[nom_ngram] - 1
                denom_count = denom_freq[denom_ngram] - 1

                if n_len == 1:
                    denom_count = sum(nom_freq.values()) - 1 
 
                p_at_n_len = 0
                if denom_count:
                    p_at_n_len = nom_count / denom_count

                c_at_n[n_len - 1] = nom_count + 1
                p_at_n[n_len - 1] = p_at_n_len 

                p_max = max(p_at_n)
                for i, p_item in enumerate(p_at_n):
                    if p_max == p_item:
                        L_counts[i] += c_at_n[i]

        total_L = sum(L_counts)
        return [L_count_item / total_L for L_count_item in L_counts]

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
            temp.extend(list(sent))

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
        return [s for s in self.text.split('\n')]

    def eval_text(self, text):
        """ Evaluates a string as per the model trained on the training text.
        """
        # Pre-process sentence given
        sents = text.split('\n')
        words = []
        for sent in sents:
            words.extend(list(sent))

        for idx, word in enumerate(words):
            if (word, ) not in self.uni_dist:
                words[idx] = TOKENS["UNK"]

        # Compute Log-Probablities
        log_prob = 0
        for ngram in nltk.ngrams(words, self.N):
            log_prob += self.eval_ngram(ngram)

        # Compute Perplexity
        num_words =  len(words)
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

            p_at_n = [0 for x in range(self.N)]

            for n_len in range(1, self.N + 1):

                nom_ngram   = tuple(ngram[len(ngram) - n_len:])
                denom_ngram = nom_ngram[:-1]


                nom_freq    = self.freqs[n_len]
                denom_freq  = self.freqs[n_len - 1 ]
                if n_len == 1:
                    nom_freq   = self.uni_dist
                    denom_freq = self.uni_dist

                nom_count   = nom_freq[nom_ngram]
                denom_count = denom_freq[denom_ngram]
                if n_len == 1:
                    denom_count = sum(nom_freq.values()) 
                
                p_at_n[n_len - 1] = 0
                if denom_count:
                    p_at_n[n_len - 1] = nom_count / denom_count 

            output_p = 0
            for index, item in enumerate(p_at_n):
                output_p += item * self.interp_coeff[index]
            return math.log2(output_p)

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










