import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    n_docs = len(documents)
    vocab = sorted(set(
        word
        for doc in documents
        for word in doc.split()
    ))
    vocabulary = {word: idx for idx, word in enumerate(vocab)}
    df = np.zeros(len(vocab), dtype=float)
    for doc in documents:
        for word in set(doc.split()):
            df[vocabulary[word]] += 1
    idf = np.log(n_docs / df)
    tfidf_matrix = np.zeros((n_docs, len(vocab)), dtype=float)
    for i, doc in enumerate(documents):
        words = doc.split()
        counts = Counter(words)
        for word, count in counts.items():
            j = vocabulary[word]
            tf = count / len(words)
            tfidf_matrix[i, j] = tf * idf[j]

    return tfidf_matrix, vocabulary