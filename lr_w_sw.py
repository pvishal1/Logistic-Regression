from __future__ import division
from __future__ import print_function
import nltk
import os
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer
import glob
import math
import numpy
import random

# 0=ham
# 1=spam

def lr():
    ham_files = len(glob.glob("train/ham/*.txt"))
    spam_files = len(glob.glob("train/spam/*.txt"))
    total_no_of_files = ham_files + spam_files
    eta = 0.01
    lamb = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_words = {}
    all_words = stem("train/*/*", all_words)
    no_of_unique_training_words = len(all_words)
    data = numpy.zeros((total_no_of_files, no_of_unique_training_words+2))
    data = populate_data("train/*/*", all_words, data, no_of_unique_training_words)

    for lb in lamb:
        pr = [random.random() for i in range(total_no_of_files)]
        w = [random.random() for i in range(no_of_unique_training_words+1)]

        for conv in range(80):
            for i in range(total_no_of_files):
                pr[i] = compute_pr(data,w,i,no_of_unique_training_words)
            dw = [0 for i in range(no_of_unique_training_words+1)]
            for j in range(no_of_unique_training_words + 1):
                for i in range(total_no_of_files):
                    dw[j] = dw[j] + (data[i][j] * (data[i][no_of_unique_training_words + 1] - pr[i]))
            for i in range(no_of_unique_training_words + 1):
                w[i] = w[i] + ((eta)*(dw[i] - ((lb)*w[i])))
        accuracy = test(w, all_words)
        print(lb," : ", accuracy)


def compute_pr(data,w,i,no_of_unique_training_words):
    sum = 0
    for j in range(no_of_unique_training_words+1):
        sum += (data[i][j]*w[j])
    try :
        pr = (math.exp(sum)/(1+math.exp(sum)))
    except OverflowError:
        pr = 0.99
    return pr

def test(w, all_words):
    filepath = glob.glob("test-2/ham/*.txt")
    test_file_count = len(filepath)
    wrong_decision = 0
    for file in filepath:
        file = file.rstrip(".txt")
        pr_o = get_prediction(file, w, all_words)
        if (pr_o >= 0) :
            wrong_decision += 1

    filepath = glob.glob("test-2/spam/*")
    test_file_count += len(filepath)
    for file in filepath:
        file = file.rstrip(".txt")
        pr_o = get_prediction(file, w, all_words)
        if (pr_o < 0):
            wrong_decision += 1
    accuracy = (test_file_count - wrong_decision) / test_file_count
    return accuracy*100

def get_prediction(file, w, all_words):
    words_count = {}
    words_count = stem(file, words_count)
    pr_o = w[0]
    for words in words_count:
        if words in all_words :
            pr_o += (w[all_words.keys().index(words)+1] * words_count[words])
    return pr_o

def stem(path, bag_of_words):
    filepath = glob.glob(path + ".txt")
    tokenizer = RegexpTokenizer("[a-zA-Z]+");
    stemmer = LancasterStemmer() # ("english")
    for file in filepath:
        if not os.path.isfile(file):
            print("File path {} does not exist. Exiting...".format(file))
            sys.exit()
        with open(file,'r') as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)
                stemmed = [stemmer.stem(t) for t in tokens]
                record_word_cnt(stemmed, bag_of_words)
    return bag_of_words

def record_word_cnt(words, bag_of_words):
    for word in words:
        if word != '':
            if word.lower() in bag_of_words:
                bag_of_words[word.lower()] += 1
            else:
                bag_of_words[word.lower()] = 1

def populate_data(path, bag_of_words, data, no_of_unique_training_words):
    filepath = glob.glob(path + ".txt")
    tokenizer = RegexpTokenizer("[a-zA-Z]+");
    stemmer = LancasterStemmer() # ("english")
    i = 0
    for file in filepath:
        word_store = {}
        if not os.path.isfile(file):
            print("File path {} does not exist. Exiting...".format(file))
            sys.exit()

        with open(file,'r') as fp:
            for line in fp:
                tokens = tokenizer.tokenize(line)
                stemmed = [stemmer.stem(t) for t in tokens]
                record_word_cnt(stemmed, word_store)
        if "train/ham/" in file:
            data[i][no_of_unique_training_words+1] = 0
        elif "train/spam/" in file:
            data[i][no_of_unique_training_words + 1] = 1
        for words in word_store:
            data[i][bag_of_words.keys().index(words)+1] = word_store[words]
        data[i][0] = 1
        i+=1
    return data
