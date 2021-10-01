import numpy as np

punc_set = ['\n', '.', ',', '\'', '?', '|', '/', '!', '_', ':']

def preprocess(text):
    for punc in punc_set:
        text = text.replace(punc, '')
    text = ' '.join(text.split()).lower()
    tokens = text.split(' ')
    return tokens

def evaluation(pred, false_symbol):
    TP, FP, FN, TN = 0, 0, 0, 0
    for obesity, label in pred:
        if obesity == True and label == 'Y':
            TP += 1
        elif obesity == True and label == false_symbol:
            FP += 1
        elif obesity == False and label == 'Y':
            FN += 1
        elif obesity == False and label == false_symbol:
            TN += 1
    accuracy = (TP+TN) / (TP+FP+TN+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1_score = (2*precision*recall) / (precision+recall)
    print("TP:", TP, ", FP:", FP, ", FN:", FN, ", TN:", TN)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1_score)
    print()