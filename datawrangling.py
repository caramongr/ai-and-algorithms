import os
from glob import glob
import textract
import nltk
import string

en_stopwords = set(nltk.corpus.stopwords.words('english'))

def create_fdist_visualizations(path):

    word_docs = glob(os.path.join(path, '*.docx'))
    text = ' '.join([textract.process(w).decode('utf-8') for w in word_docs])
    # remove punctuation, numbers, stopwords
    translator = str.maketrans('', '', string.punctuation + string.digits)
    text = text.translate(translator)
    words = text.lower().split()
    words = [w for w in words if w not in en_stopwords and len(w) > 3]
    unigram_fd = nltk.FreqDist(words)
    bigrams = list([' '.join(bg) for bg in nltk.bigrams(words)])
    bigram_fd = nltk.FreqDist(bigrams)
    unigram_fd.plot(20)


create_fdist_visualizations('data/gfsr_docs/docx/')