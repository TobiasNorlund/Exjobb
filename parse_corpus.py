
# Parses a corpus to for each word output a matrix of size (2k) x d, where k is the half context window size and d is
# the dimensionality of the index vectors

from Dictionary import Dictionary
from OneBCorpus import OneBCorpus
from scipy.sparse import csr_matrix
import numpy as np
import random
import numpy.random as nprnd
import sys

def parse(corpus, k, d, epsilon):

    win_size = 2*k+1

    index_dict = dict()
    context_dict = dict()

    window = [None] * win_size

    # Create generator
    word_gen = corpus.read_word()

    # Read in first window
    for wi in range(2*k):
        window[wi] = word = next(word_gen)
        if word not in index_dict:
            index_dict[word] = gen_index_vector(d, epsilon)
            context_dict[word] = np.zeros((2*k,d), dtype=np.int32)

    wi = 2*k # window index
    win_idx = np.array([i for i in range(-k,k+1) if i != 0])

    print("Parsing corpus...")

    for word in word_gen:

        window[wi] = word

        # Focus word
        fwi = (wi - k) % win_size
        fword = window[fwi]

        if word not in index_dict:
            index_dict[word] = gen_index_vector(d, epsilon)
            context_dict[word] = np.zeros((2*k,d), dtype=np.int32)

        for j in range(2*k): #j in (fwi + win_idx) % win_size:
            wj = (fwi + win_idx[j]) % win_size
            context_dict[fword][j,] += index_dict[window[wj]]

        wi = (wi+1) % win_size
        print_progress(corpus)


    return context_dict


def gen_index_vector(d, epsilon):
    data = [random.choice([-1, 1]) for _ in range(epsilon)]
    idx = nprnd.choice(d, epsilon, False) #.randint(d, size=epsilon)
    return csr_matrix((data, (np.zeros(epsilon), idx)), dtype=np.int8, shape=(1,d))

def print_progress(corpus):
    #sys.stdout.write("\rProgress %i" % corpus.get_progress())
    sys.stdout.write("\r %.2f%%" % (corpus.get_progress()*100))

class DummyCorpus(object):

    def read_word(self):
        for word in ["the", "cat", "sat", "on", "the", "mat", "and", "sat"]:
            yield word

if __name__ == "__main__":
    corpus = OneBCorpus('1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100')
    contexts = parse(corpus, 2, 2000, 10)

    print("Done")