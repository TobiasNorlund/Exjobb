import os

class OneBCorpus(object):

    def __init__(self, filepath):
        self.filepath = filepath

    def read_word(self):
        with open(self.filepath) as f:
            self.total_size = os.path.getsize(self.filepath)
            self.total_read = 0.0
            for line in f:
                for word in line.split():
                    yield word
                    self.total_read += len(word)+1.0

    def get_progress(self):
        return self.total_read / self.total_size

if __name__ == "__main__":

    corpus = OneBCorpus('1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100')

    print("Done!")