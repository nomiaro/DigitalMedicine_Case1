import numpy as np
import os
import csv
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
from sklearn.model_selection import GridSearchCV
def load():
    texts, train_texts, val_texts, train_labels = [], [], [], []
    train_root = "../data/Train_Textual/"
    val_root = "../data/Validation/"
    test_root = "../data/Test_Intuitive/"
    for filename in os.listdir(train_root):
        text = open(train_root+filename).read()
        text = preprocess(text)
        texts.append(text)
        train_texts.append(text)
        train_labels.append(filename[0])
    for filename in os.listdir(test_root):
        text = open(test_root+filename).read()
        text = preprocess(text)
        texts.append(text)
        train_texts.append(text)
        if filename[0] == 'N': train_labels.append('U')
        else: train_labels.append(filename[0])
    for filename in os.listdir(val_root):
        text = open(val_root+filename).read()
        text = preprocess(text)
        texts.append(text)
        val_texts.append(text)
    
    return texts, train_texts, val_texts, train_labels

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

"""
def NB(train_data_Tfidf, val_data_Tfidf, train_labels):
    naive = naive_bayes.MultinomialNB()
    naive.fit(train_data_Tfidf, train_labels)
    pred = naive.predict(val_data_Tfidf)
    count = 0
    with open('nb.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Obesity'])
        val_root = "../data/Validation/" 
        for filename in os.listdir(val_root):
            writer.writerow([filename, pred[count]])
            count += 1
"""     

def SVM(train_data_Tfidf, val_data_Tfidf, train_labels):
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_data_Tfidf, train_labels)
    # 用自己test
    pred = SVM.predict(train_data_Tfidf)
    print("SVM accuracy:", accuracy_score(pred, train_labels))
    print("SVM f1 score:", f1_score(pred, train_labels))
    # val
    pred = SVM.predict(val_data_Tfidf)
    count = 0
    with open('svm_TfidfVectorizertuning.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Obesity'])
        val_root = "../data/Validation/" 
        for filename in os.listdir(val_root):
            writer.writerow([filename, pred[count]])
            count += 1
      
def training(texts, train_texts, val_texts, train_labels):
    Encoder = LabelEncoder()
    train_labels = Encoder.fit_transform(train_labels)
    Tfidf_vect = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    Tfidf_vect.fit(texts)
    train_data_Tfidf = Tfidf_vect.transform(train_texts)
    val_data_Tfidf = Tfidf_vect.transform(val_texts)
    print(Tfidf_vect.get_feature_names_out())
    #NB(train_data_Tfidf, val_data_Tfidf, train_labels)
    SVM(train_data_Tfidf, val_data_Tfidf, train_labels)

if __name__ == '__main__':
    texts, train_texts, val_texts, train_labels = load()
    training(texts, train_texts, val_texts, train_labels)
