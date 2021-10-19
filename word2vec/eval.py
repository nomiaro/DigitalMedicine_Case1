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
from gensim.models.word2vec import Word2Vec
import gensim
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
def get_word_vector(model, content):

    
    # vec = np.zeros(2).reshape((1, 2))
    vec = np.zeros(50).reshape((1, 50))
    count = 0
    #words = remove_some(words)
    for word in content[1:]:
        try:
            count += 1
            # vec += model[word].reshape((1, 2))
            vec += model.wv[word].reshape((1, 50))
            # print(vec)
        except KeyError:
            continue
    vec /= count
    return vec
def SVM(train_data_Tfidf, val_data_Tfidf, train_labels):
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(train_data_Tfidf, train_labels)
    pred = SVM.predict(val_data_Tfidf)
    count = 0
    with open('svm_word2vec.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Obesity'])
        val_root = "../data/Validation/" 
        for filename in os.listdir(val_root):
            writer.writerow([filename, pred[count]])
            count += 1
      
def training(texts, train_texts, val_texts, train_labels):
    Encoder = LabelEncoder()
    train_labels = Encoder.fit_transform(train_labels)
    """
    Tfidf_vect = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    Tfidf_vect.fit(texts)
    train_data_Tfidf = Tfidf_vect.transform(train_texts)
    val_data_Tfidf = Tfidf_vect.transform(val_texts)
    """
    x_train = []
    val = []
    if os.path.exists("./model"):
        model = Word2Vec.load('./model')
    else:
        model = Word2Vec(texts, min_count=1, vector_size=50)  # 訓練skip-gram模型
        model.save("./model")
    for doc in train_texts:
        x_train.append(get_word_vector(model, doc))
    for doc in val_texts:
        val.append(get_word_vector(model, doc))
    #NB(x_train, train_label, x_test, test_label)
    x_train = np.array(x_train)
    x_train = x_train.squeeze()
    val = np.array(val)
    val = val.squeeze()
    #print(Tfidf_vect.get_feature_names_out())
    #NB(train_data_Tfidf, val_data_Tfidf, train_labels)
    SVM(x_train, val, train_labels)

if __name__ == '__main__':
    texts, train_texts, val_texts, train_labels = load()
    training(texts, train_texts, val_texts, train_labels)
