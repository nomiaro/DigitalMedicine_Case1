import numpy as np
import sys
import os
from utils import *

word_bag = ['obese', 'obesity', 'overweight']

def inference(tokens):
    for token in tokens:
        if token in word_bag:
            return True
    return False
    
def run(root, mode):
    pred = []
    for filename in os.listdir(root):
        text = open(root+filename).read()
        tokens = preprocess(text)
        label = filename[0]
        obesity = inference(tokens)
        pred.append((obesity, label))

    if mode == 'Train':
        print("Train Textual Evaluation")
        evaluation(pred, 'U')
    elif mode == 'Test':
        print("Test Intuitive Evaluation")
        evaluation(pred, 'N')

if __name__ == '__main__':
    run("./data/Train_Textual/", 'Train')
    run("./data/Test_Intuitive/", 'Test')
