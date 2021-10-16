import numpy as np
import csv
import os
from nltk.tokenize import word_tokenize

word_bag = ['obese', 'obesity', 'overweight', 'Obese', 'Obesity', 'Overweight']

def inference(tokens):
    for token in tokens:
        if token in word_bag:
            return 1
    return 0
    
def run(root):
    with open('submission_nltk.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Obesity'])
        for filename in os.listdir(root):
            text = open(root+filename).read()
            tokens = word_tokenize(text)
            obesity = inference(tokens)
            writer.writerow([filename, obesity])
    
if __name__ == '__main__':
    run("../data/Validation/")
