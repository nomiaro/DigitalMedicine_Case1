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

def training(texts, labels):
    train_data, test_data, train_label, test_label = texts, texts, labels, labels
    Encoder = LabelEncoder()
    train_label = Encoder.fit_transform(train_label)
    test_label =  Encoder.fit_transform(test_label)
    """
    Tfidf_vect = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
    Tfidf_vect.fit(texts)
    train_data_Tfidf = Tfidf_vect.transform(train_data)
    test_data_Tfidf = Tfidf_vect.transform(test_data)
    """
    x_train = []
    x_test = []
    if os.path.exists("./doc2vec_model"):
        model = gensim.models.Doc2Vec.load('./doc2vec_model')
    else:
        texts = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
        model = gensim.models.Doc2Vec(texts, vector_size=200, window=5, min_count=1)
        model.save("./doc2vec_model")
    print(len(model.dv))
    for idx, docvec in enumerate(model.dv):
            if idx < 800:
                x_train.append(docvec)
                x_test.append(docvec)
            if idx == 799:
                break
    #NB(x_train, train_label, x_test, test_label)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.squeeze()
    x_test = x_test.squeeze()
    SVM(x_train, train_label, x_test, test_label)

if __name__ == '__main__':
    np.random.seed(500)
    texts, labels = load()
    training(texts, labels)
