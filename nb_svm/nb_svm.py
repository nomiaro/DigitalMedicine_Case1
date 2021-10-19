import numpy as np
import os
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, f1_score
from gensim.models.word2vec import Word2Vec
def load():
    train_root = "../data/Train_Textual/"
    test_root = "../data/Test_Intuitive/"
    texts, labels = [], []
    for filename in os.listdir(train_root):
        text = open(train_root+filename).read()
        texts.append(preprocess(text))
        labels.append(filename[0])
    for filename in os.listdir(test_root):
        text = open(test_root+filename).read()
        texts.append(preprocess(text))
        if filename[0] == 'N': labels.append('U')
        else: labels.append(filename[0])
    return texts, labels

def preprocess(text):
    # tokenize
    tokens = word_tokenize(text.lower())
    # get tag_map
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    final_tokens = []
    word_Lemmatized = WordNetLemmatizer()
    for token, tag in pos_tag(tokens):
        if token not in stopwords.words('english') and token.isalpha():
            token = word_Lemmatized.lemmatize(token, tag_map[tag[0]])
            final_tokens.append(token)
    return str(final_tokens)

def NB(train_data_Tfidf, train_label, test_data_Tfidf, test_label):
    naive = naive_bayes.MultinomialNB()
    naive.fit(train_data_Tfidf, train_label)

    pred = naive.predict(test_data_Tfidf)
    print("NB accuracy:", accuracy_score(pred, test_label))
    print("NB f1 score:", f1_score(pred, test_label))

def SVM(train_data_Tfidf, train_label, test_data_Tfidf, test_label):
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_data_Tfidf, train_label)

    pred = SVM.predict(test_data_Tfidf)
    print("SVM accuracy:", accuracy_score(pred, test_label))
    print("SVM f1 score:", f1_score(pred, test_label))

def training(texts, labels):
    train_data, test_data, train_label, test_label = model_selection.train_test_split(texts, labels, test_size=0.1)
    Encoder = LabelEncoder()
    train_label = Encoder.fit_transform(train_label)
    test_label =  Encoder.fit_transform(test_label)
    Tfidf_vect = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    Tfidf_vect.fit(texts)
    train_data_Tfidf = Tfidf_vect.transform(train_data)
    test_data_Tfidf = Tfidf_vect.transform(test_data)
    NB(train_data_Tfidf, train_label, test_data_Tfidf, test_label)
    SVM(train_data_Tfidf, train_label, test_data_Tfidf, test_label)

if __name__ == '__main__':
    np.random.seed(500)
    texts, labels = load()
    training(texts, labels)
