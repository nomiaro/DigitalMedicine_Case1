import numpy as np
import os
from nltk.tokenize import word_tokenize

word_bag = ['obese', 'obesity', 'overweight']

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

def inference(tokens):
    for token in tokens:
        if token in word_bag:
            return True
    return False
    
def run(root, mode):
    pred = []
    for filename in os.listdir(root):
        text = open(root+filename).read()
        tokens = word_tokenize(text.lower())
        obesity = inference(tokens)
        #if obesity == True and filename[0] == 'U': print(filename)
        pred.append((obesity, filename[0]))
    
    if mode == 'Train':
        print("Train Textual Evaluation")
        evaluation(pred, 'U')
    elif mode == 'Test':
        print("Test Intuitive Evaluation")
        evaluation(pred, 'N')
    
if __name__ == '__main__':
    run("../data/Train_Textual/", 'Train')
    #run("../data/Test_Intuitive/", 'Test')
    
