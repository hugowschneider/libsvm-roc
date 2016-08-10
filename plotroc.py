#!/usr/bin/env python
# This tool allow users to plot SVM-prob ROC curve from data
from svmutil import *
from sys import argv, platform
from os import path, popen
from random import randrange, seed
from operator import itemgetter
from time import sleep
import numpy as np
import matplotlib.pyplot as plt

# search path for gnuplot executable
# be careful on using windows LONG filename, surround it with double quotes.
# and leading 'r' to make it raw string, otherwise, repeat \\.


def get_pos_deci(model_file, test_y, test_x):
    model = svm_load_model(model_file)
    # predict and grab decision value, assure deci>0 for label+,
    # the positive descision value = val[0]*labels[0]
    labels = model.get_labels()
    py, evals, deci = svm_predict(test_y, test_x, model, '-b 1')
    deci = [labels[0] * val[0] for val in deci]
    return deci, model


# get_cv_deci(prob_y[], prob_x[], svm_parameter param, nr_fold)
# input raw attributes, labels, param, cv_fold in decision value building
# output list of decision value, remember to seed(0)
def get_cv_deci(prob_y, prob_x, param, nr_fold):
    if nr_fold == 1 or nr_fold == 0:
        deci, model = get_pos_deci(prob_y, prob_x, prob_y, prob_x, param)
        return deci
    deci, model = [], []
    prob_l = len(prob_y)

    # random permutation by swapping i and j instance
    for i in range(prob_l):
        j = randrange(i, prob_l)
        prob_x[i], prob_x[j] = prob_x[j], prob_x[i]
        prob_y[i], prob_y[j] = prob_y[j], prob_y[i]

    # cross training : folding
    for i in range(nr_fold):
        begin = i * prob_l // nr_fold
        end = (i + 1) * prob_l // nr_fold
        train_x = prob_x[:begin] + prob_x[end:]
        train_y = prob_y[:begin] + prob_y[end:]
        test_x = prob_x[begin:end]
        test_y = prob_y[begin:end]
        subdeci, submdel = get_pos_deci(train_y, train_x, test_y, test_x, param)
        deci += subdeci
    return deci


# processing argv and set some global variables
def proc_argv(argv=argv):
    # print("Usage: %s " % argv[0])
    # The command line : ./plotroc.py [-v cv_fold | -T testing_file] [libsvm-options] training_file
    parts = chunks(argv[2:],3)

    model_files = []
    test_files = []
    titles = []

    for i in parts:
        test_files.append(i[0])
        model_files.append(i[1])
        titles.append(i[2])

    return model_files, test_files, titles


def plot_roc(deci, label, title):
    # count of postive and negative labels
    db = []
    pos, neg = 0, 0
    for i in range(len(label)):
        if label[i] > 0:
            pos += 1
        else:
            neg += 1
        db.append([deci[i], label[i]])

    # sorting by decision value
    db = sorted(db, key=itemgetter(0), reverse=True)

    # calculate ROC
    xy_arr = []
    x_arr = []
    y_arr = []
    tp, fp = 0., 0.  # assure float division
    for i in range(len(db)):
        if db[i][1] > 0:  # positive
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])
        x_arr.append(fp / neg)
        y_arr.append(tp / pos)

    # area under curve
    aoc = 0.
    prev_x = 0
    for x, y in xy_arr:
        if x != prev_x:
            aoc += (x - prev_x) * y
            prev_x = x

    plt.plot(x_arr, y_arr, label=title % (aoc * 100))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main():
    if len(argv) < 5 or (len(argv) - 2) % 3 != 0:
        print("Usage: %s plot_title testing_file1 model_file1 title1 [testing_file2 model_file2 title2] ... [testing_fileN model_fileN titleN]" % argv[0])
        raise SystemExit
    model_file, test_file, title = proc_argv()
    # read data

    plt.figure()
    for i,m in enumerate(model_file):
        test_y, test_x = svm_read_problem(test_file[i])
        if set(test_y) != {1, -1}:
            print("ROC is only applicable to binary classes with labels 1, -1")
            raise SystemExit
        deci, model = get_pos_deci(model_file[i], test_y, test_x)
        plot_roc(deci, test_y, title[i])

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(argv[1])
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    main()