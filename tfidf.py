import math
from nltk.tokenize import word_tokenize

corpus = [
    "Menna love studying IR.",
    "IR is not for everyone.",
    "IR class is full of students.",
    "Studying IR is enjoyable."
]

def TF(word, document):
    words = word_tokenize(document.lower())
    #words = document.split()
    word_count = 0
    total_words = len(words)
    for w in words:
        if w == word:
            word_count += 1
    tf = word_count / total_words
    return math.log10(tf + 1)

def IDF(word, corpus):
    total_documents = len(corpus)
    document_count = 0
    for doc in corpus:
        if word in doc:
            document_count += 1
    return math.log10(total_documents / (document_count+1))

def tfidf(corpus):
    tfidf_scores = {}
    for document in corpus:
        words = word_tokenize(document.lower())
        for word in set(words):
            tf = TF(word, document)
            idf = IDF(word, corpus)
            tfidf_scores[word] = tf * idf